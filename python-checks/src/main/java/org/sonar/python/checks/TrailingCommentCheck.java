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
package org.sonar.python.checks;

import java.util.List;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S139")
public class TrailingCommentCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT_LEGAL_COMMENT_PATTERN = "^#\\s*+([^\\s]++|fmt.*|type.*|noqa.*)$";
  private static final String MESSAGE = "Move this trailing comment on the previous empty line.";

  @RuleProperty(
    key = "legalTrailingCommentPattern",
    description = "Pattern for text of trailing comments that are allowed. By default, Mypy and Black pragma comments as well as comments containing only one word.",
    defaultValue = DEFAULT_LEGAL_COMMENT_PATTERN)
  public String legalCommentPattern = DEFAULT_LEGAL_COMMENT_PATTERN;

  private int previousTokenLine;

  private List<String> lines;

  @Override
  public void initialize(Context context) {
    Pattern pattern = Pattern.compile(legalCommentPattern);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      previousTokenLine = -1;
      lines = null;
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
      Token token = (Token) ctx.syntaxNode();
      for (Trivia trivia : token.trivia()) {
        Token commentToken = trivia.token();
        if (previousTokenLine == commentToken.line()) {
          String comment = commentToken.value();
          if (!pattern.matcher(comment).matches()) {
            var issue = ctx.addIssue(commentToken, MESSAGE);
            String line = getLines(ctx).get(commentToken.line() - 1);
            addQuickFix(issue, commentToken, line);
          }
        }
      }
      previousTokenLine = token.line();
    });
  }

  private static void addQuickFix(PreciseIssue issue, Token commentToken, String line) {
    String indent = calculateIndent(line);
    PythonTextEdit insertComment = TextEditUtils.insertAtPosition(commentToken.line(), 0, indent + commentToken.value() + "\n");

    int startColumnRemove = calculateStartColumnToRemove(commentToken, line);
    PythonTextEdit removeTrailingComment = TextEditUtils.removeRange(commentToken.line(), startColumnRemove, commentToken.line(), line.length());

    PythonQuickFix fix = PythonQuickFix.newQuickFix(MESSAGE, removeTrailingComment, insertComment);
    issue.addQuickFix(fix);
  }

  private static String calculateIndent(String line) {
    String lineWithoutIndent = line.stripLeading();
    int column = line.indexOf(lineWithoutIndent);
    return " ".repeat(column);
  }

  private List<String> getLines(SubscriptionContext ctx) {
    if (lines == null) {
      lines = ctx.pythonFile().content().lines().toList();
    }
    return lines;
  }

  private static int calculateStartColumnToRemove(Token commentToken, String line) {
    return line.substring(0, commentToken.column()).stripTrailing().length();
  }
}

