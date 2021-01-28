/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
  private static final List<Tree.Kind> EXECUTABLE_LINES = Arrays.asList(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.COMPOUND_ASSIGNMENT, Tree.Kind.EXPRESSION_STMT, Tree.Kind.IMPORT_STMT,
    Tree.Kind.IMPORT_NAME, Tree.Kind.IMPORT_FROM, Tree.Kind.CONTINUE_STMT, Tree.Kind.BREAK_STMT, Tree.Kind.YIELD_STMT, Tree.Kind.RETURN_STMT, Tree.Kind.PRINT_STMT,
    Tree.Kind.PASS_STMT, Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT, Tree.Kind.IF_STMT, Tree.Kind.RAISE_STMT, Tree.Kind.TRY_STMT, Tree.Kind.EXCEPT_CLAUSE,
    Tree.Kind.EXEC_STMT, Tree.Kind.ASSERT_STMT, Tree.Kind.DEL_STMT, Tree.Kind.GLOBAL_STMT, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT);

  private Set<Integer> noSonar = new HashSet<>();
  private Set<Integer> linesOfCode = new HashSet<>();
  private Set<Integer> linesOfComments = new HashSet<>();
  private Set<Integer> linesOfDocstring = new HashSet<>();
  private Set<Integer> executableLines = new HashSet<>();
  private int statements = 0;
  private int classDefs = 0;

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

  private void handleDocString(@Nullable StringLiteral docstring) {
    if (docstring != null) {
      for (Tree stringElement : docstring.children()) {
        TokenLocation location = new TokenLocation(stringElement.firstToken());
        for (int line = location.startLine(); line <= location.endLine(); line++) {
          linesOfDocstring.add(line);
        }
      }
    }
  }

  /**
   * Gets the lines of codes and lines of comments (with character #).
   * Does not get the lines of docstrings.
   */
  private void visitToken(Token token) {
    if (token.type().equals(GenericTokenType.EOF)) {
      return;
    }

    if (!token.type().equals(PythonTokenType.DEDENT) && !token.type().equals(PythonTokenType.INDENT) && !token.type().equals(PythonTokenType.NEWLINE)) {
      // Handle all the lines of the token
      String[] tokenLines = token.value().split("\n", -1);
      int tokenLine = token.line();
      for (int line = tokenLine; line < tokenLine + tokenLines.length; line++) {
        linesOfCode.add(line);
      }
    }

    for (Trivia trivia : token.trivia()) {
      visitComment(trivia);
    }
  }

  private void visitComment(Trivia trivia) {
    String commentLine = getContents(trivia.token().value());
    int line = trivia.token().line();
    if (commentLine.contains("NOSONAR")) {
      linesOfComments.remove(line);
      noSonar.add(line);
    } else if (!isBlank(commentLine)) {
      linesOfComments.add(line);
    }
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
    return Collections.unmodifiableSet(executableLines);
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

  public int getStatements() {
    return statements;
  }

  public int getClassDefs() {
    return classDefs;
  }
}
