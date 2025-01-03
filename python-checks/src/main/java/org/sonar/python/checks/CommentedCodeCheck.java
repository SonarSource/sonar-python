/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

@Rule(key = "S125")
public class CommentedCodeCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Remove this commented out code.";
  // Regex coming from https://www.python.org/dev/peps/pep-0263/#defining-the-encoding
  private static final Pattern ENCODING_PATTERN = Pattern.compile(".*?coding[:=][ \\t]*([-_.a-zA-Z0-9]+)\n");
  private static final Pattern SINGLE_WORD_PATTERN = Pattern.compile("\\s*[\\w/\\-]+\\s*#*\n*");
  private static final Pattern IS_EMPTY_PATTERN = Pattern.compile("\\s*");

  private static final String DEFAULT_EXCEPTION_PATTERN = "(fmt|py\\w+):.*";
  private static final Pattern DATABRICKS_MAGIC_COMMAND_PATTERN = Pattern.compile("^\\h*(MAGIC|COMMAND).*");
  private static final PythonParser parser = PythonParser.create();

  private Pattern exceptionPattern;

  @RuleProperty(
    key = "exception",
    description = "Regular expression used to ignore commented out code. Only a full match is excluded.",
    defaultValue = "" + DEFAULT_EXCEPTION_PATTERN)
  public String exception = DEFAULT_EXCEPTION_PATTERN;

  @Override
  public void initialize(Context context) {
    exceptionPattern = Pattern.compile(exception);

    context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
      Token token = (Token) ctx.syntaxNode();
      List<List<Trivia>> groupedTrivias = groupTrivias(token);
      for (List<Trivia> triviaGroup : groupedTrivias) {
        checkTriviaGroup(triviaGroup, ctx);
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.STRING_LITERAL, ctx -> {
      StringLiteral stringLiteral = (StringLiteral) ctx.syntaxNode();
      if (isMultilineComment(stringLiteral)) {
        visitMultilineComment(stringLiteral, ctx);
      }
    });
  }

  private static boolean isMultilineComment(StringLiteral stringLiteral) {
    Tree parent = stringLiteral.parent();
    StringElement firstElement = stringLiteral.stringElements().get(0);
    return firstElement.isTripleQuoted() && parent.is(Tree.Kind.EXPRESSION_STMT);
  }

  private static void visitMultilineComment(StringLiteral stringLiteral, SubscriptionContext ctx) {
    String text = Expressions.unescape(stringLiteral);
    text = text.trim();
    if (!isEmpty(text) && isTextParsedAsCode(text)) {
      ctx.addIssue(stringLiteral, MESSAGE);
    }
  }

  private void checkTriviaGroup(List<Trivia> triviaGroup, SubscriptionContext ctx) {
    String text = getTextForParsing(triviaGroup);
    if (isEmpty(text)) {
      return;
    }
    if (isTextParsedAsCode(text) && !isEncoding(triviaGroup.get(0), text)) {
      ctx.addIssue(triviaGroup.get(0).token(), MESSAGE);
    }
  }

  private String getTextForParsing(List<Trivia> triviaGroup) {
    StringBuilder commentTextSB = new StringBuilder();
    for (Trivia trivia : triviaGroup) {
      String value = trivia.value();
      while (value.startsWith("#") || value.startsWith(" #")) {
        value = value.substring(1);
      }
      if (value.startsWith(" ")) {
        value = value.substring(1);
      }
      if (triviaGroup.size() == 1) {
        value = value.trim();
      }
      if (!(isOneWord(value) || isException(value))) {
        commentTextSB.append(value);
        commentTextSB.append("\n");
      }
    }
    return commentTextSB.toString();
  }

  private boolean isException(String text) {
    boolean isDatabricksMagicCommand = DATABRICKS_MAGIC_COMMAND_PATTERN.matcher(text).matches();
    return exceptionPattern.matcher(text).matches() || isDatabricksMagicCommand;
  }

  private static boolean isOneWord(String text) {
    return SINGLE_WORD_PATTERN.matcher(text).matches();
  }

  private static boolean isEmpty(String text) {
    return IS_EMPTY_PATTERN.matcher(text).matches();
  }

  // "source code encoding" comments (e.g. # coding=utf8) should be excluded
  // Note that encoding must be on line 1 or 2
  private static boolean isEncoding(Trivia trivia, String text) {
    return trivia.token().line() < 3 && ENCODING_PATTERN.matcher(text).matches();
  }

  private static boolean isTextParsedAsCode(String text) {
    try {
      AstNode astNode = parser.parse(text);
      FileInput parse = new PythonTreeMaker().fileInput(astNode);
      return parse.statements() != null && !isSimpleExpression(parse);
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean isSimpleExpression(FileInput fileInput) {
    if (fileInput.statements().statements().size() > 1) {
      return false;
    }
    Statement statement = fileInput.statements().statements().get(0);
    return statement.is(Tree.Kind.EXPRESSION_STMT) || statement.is(Tree.Kind.ANNOTATED_ASSIGNMENT);
  }

  private static List<List<Trivia>> groupTrivias(Token token) {
    List<List<Trivia>> result = new ArrayList<>();
    List<Trivia> currentGroup = null;
    for (Trivia trivia : token.trivia()) {
      currentGroup = handleOneLineComment(result, currentGroup, trivia);
    }
    if (currentGroup != null) {
      result.add(currentGroup);
    }
    return result;
  }

  private static List<Trivia> handleOneLineComment(List<List<Trivia>> result, @Nullable List<Trivia> currentGroup, Trivia trivia) {
    List<Trivia> newTriviaGroup = currentGroup;
    if (currentGroup == null) {
      newTriviaGroup = new ArrayList<>();
      newTriviaGroup.add(trivia);
    } else if (currentGroup.get(currentGroup.size() - 1).token().line() + 1 == trivia.token().line()) {
      newTriviaGroup.add(trivia);
    } else {
      result.add(currentGroup);
      newTriviaGroup = new ArrayList<>();
      newTriviaGroup.add(trivia);
    }
    return newTriviaGroup;
  }
}
