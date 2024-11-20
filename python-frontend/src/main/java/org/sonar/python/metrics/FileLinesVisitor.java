/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.metrics;

import com.sonar.sslr.api.GenericTokenType;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TokenLocation;
import org.sonar.python.api.PythonTokenType;

/**
 * Visitor that computes {@link CoreMetrics#NCLOC_DATA_KEY} and {@link CoreMetrics#COMMENT_LINES} metrics used by the DevCockpit.
 */
public class FileLinesVisitor extends PythonSubscriptionCheck {

  /**
   * Tree.Kind.ELSE_CLAUSE is not in this list to avoid counting else: lines as executables.
   * This is to replicate behavior of some python coverage tools (like what is done by coveralls).
   */
  private static final List<Tree.Kind> EXECUTABLE_LINES = Arrays.asList(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.COMPOUND_ASSIGNMENT, Tree.Kind.EXPRESSION_STMT,
    Tree.Kind.IMPORT_NAME, Tree.Kind.IMPORT_FROM, Tree.Kind.CONTINUE_STMT, Tree.Kind.BREAK_STMT, Tree.Kind.YIELD_STMT, Tree.Kind.RETURN_STMT, Tree.Kind.PRINT_STMT,
    Tree.Kind.PASS_STMT, Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT, Tree.Kind.IF_STMT, Tree.Kind.RAISE_STMT, Tree.Kind.TRY_STMT, Tree.Kind.EXCEPT_CLAUSE,
    Tree.Kind.EXEC_STMT, Tree.Kind.ASSERT_STMT, Tree.Kind.DEL_STMT, Tree.Kind.GLOBAL_STMT, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT);

  private final Set<Integer> noSonar = new HashSet<>();
  private final Set<Integer> linesOfCode = new HashSet<>();
  private final Set<Integer> linesOfComments = new HashSet<>();
  private final Set<Integer> linesOfDocstring = new HashSet<>();
  private final Set<Integer> executableLines = new HashSet<>();
  private final boolean isNotebook;
  private int statements = 0;
  private int classDefs = 0;

  public FileLinesVisitor(boolean isNotebook) {
    this.isNotebook = isNotebook;
  }

  public FileLinesVisitor() {
    this(false);
  }

  @Override
  public void scanFile(PythonVisitorContext visitorContext) {
    SubscriptionVisitor.analyze(Collections.singleton(this), visitorContext);
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> visitFile());
    EXECUTABLE_LINES.forEach(kind -> context.registerSyntaxNodeConsumer(kind, this::visitNode));
    context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> visitToken((Token) ctx.syntaxNode()));
  }

  private void visitFile() {
    noSonar.clear();
    linesOfCode.clear();
    linesOfComments.clear();
    linesOfDocstring.clear();
    executableLines.clear();
  }

  private void visitNode(SubscriptionContext ctx) {
    Tree tree = ctx.syntaxNode();
    if (tree.is(Tree.Kind.FILE_INPUT)) {
      handleDocString(((FileInput) tree).docstring());
    } else {
      statements++;
      executableLines.add(tree.firstToken().line());
    }
    if (tree.is(Tree.Kind.CLASSDEF)) {
      classDefs++;
      handleDocString(((ClassDef) tree).docstring());
    }
    if (tree.is(Tree.Kind.FUNCDEF)) {
      handleDocString(((FunctionDef) tree).docstring());
    }
  }

  protected void handleDocString(@Nullable StringLiteral docstring) {
    linesOfDocstring.addAll(countDocstringLines(docstring));
  }

  public static Set<Integer> countDocstringLines(@Nullable StringLiteral docstring) {
    Set<Integer> lines = new HashSet<>();
    if (docstring != null) {
      for (Tree stringElement : docstring.children()) {
        TokenLocation location = new TokenLocation(stringElement.firstToken());
        for (int line = location.startLine(); line <= location.endLine(); line++) {
          lines.add(line);
        }
      }
    }
    return lines;
  }

  /**
   * Gets the lines of codes and lines of comments (with character #).
   * Does not get the lines of docstrings.
   */
  private void visitToken(Token token) {
    if (token.type().equals(GenericTokenType.EOF)) {
      return;
    }

    linesOfCode.addAll(tokenLineNumbers(token));

    for (Trivia trivia : token.trivia()) {
      visitComment(trivia, token);
    }
  }

  public static Set<Integer> tokenLineNumbers(Token token) {
    Set<Integer> lines = new HashSet<>();
    if (!token.type().equals(PythonTokenType.DEDENT) && !token.type().equals(PythonTokenType.INDENT) && !token.type().equals(PythonTokenType.NEWLINE)) {
      // Handle all the lines of the token
      String[] tokenLines = token.value().split("\n", -1);
      int tokenLine = token.pythonLine();
      for (int line = tokenLine; line < tokenLine + tokenLines.length; line++) {
        lines.add(line);
      }
    }
    return lines;
  }

  private void visitComment(Trivia trivia, Token parentToken) {
    String commentLine = getContents(trivia.token().value());
    int line = trivia.token().line();
    if (containsNoSonarComment(trivia)) {
      linesOfComments.remove(line);
      addNoSonarLines(trivia, parentToken);
    } else if (!isBlank(commentLine)) {
      linesOfComments.add(line);
    }
  }

  public static boolean containsNoSonarComment(Trivia trivia) {
    String commentLine = getContents(trivia.token().value());
    return commentLine.contains("NOSONAR");
  }

  @Override
  public void leaveFile() {
    // account for the docstring lines
    for (Integer line : linesOfDocstring) {
      executableLines.remove(line);
      linesOfCode.remove(line);
      linesOfComments.add(line);
    }
  }

  public Set<Integer> getLinesWithNoSonar() {
    return Collections.unmodifiableSet(noSonar);
  }

  public Set<Integer> getLinesOfCode() {
    return Collections.unmodifiableSet(linesOfCode);
  }

  public int getCommentLineCount() {
    return linesOfComments.size();
  }

  public Set<Integer> getExecutableLines() {
    return isNotebook ? Set.of() : Collections.unmodifiableSet(executableLines);
  }

  private static boolean isBlank(String line) {
    for (int i = 0; i < line.length(); i++) {
      if (Character.isLetterOrDigit(line.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  private static String getContents(String comment) {
    // Comment always starts with "#"
    return comment.substring(comment.indexOf('#'));
  }

  private void addNoSonarLines(Trivia trivia, Token parentToken) {
    int line = trivia.token().line();
    if (parentToken.parent().is(Tree.Kind.EXPRESSION_STMT)) {
      ExpressionStatement expressionStatement = (ExpressionStatement) parentToken.parent();
      if (!expressionStatement.expressions().isEmpty() && expressionStatement.expressions().get(0).is(Tree.Kind.STRING_LITERAL)) {
        // Count every line of a string literal as part of the "NOSONAR" scope
        StringLiteral stringLiteral = (StringLiteral) expressionStatement.expressions().get(0);
        int firstLine = stringLiteral.firstToken().line();
        for (int i = firstLine; i < line + 1; i++) {
          noSonar.add(i);
        }
        return;
      }
    }
    noSonar.add(line);
  }

  public int getStatements() {
    return statements;
  }

  public int getClassDefs() {
    return classDefs;
  }
}
