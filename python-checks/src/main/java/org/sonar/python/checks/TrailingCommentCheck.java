/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.checks;

import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;

@Rule(key = "S139")
public class TrailingCommentCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT_LEGAL_COMMENT_PATTERN = "^#\\s*+([^\\s]++|fmt.*|type.*)$";
  private static final String MESSAGE = "Move this trailing comment on the previous empty line.";
  private static final Pattern NON_WHITESPACE_PATTERN = Pattern.compile("\\S");

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
      lines = ctx.pythonFile().content().lines().collect(Collectors.toList());
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
      Token token = (Token) ctx.syntaxNode();
      for (Trivia trivia : token.trivia()) {
        if (previousTokenLine == trivia.token().line()) {
          String comment = trivia.token().value();
          if (!pattern.matcher(comment).matches()) {
            IssueWithQuickFix issue = (IssueWithQuickFix) ctx.addIssue(trivia.token(), MESSAGE);
            addQuickFix(issue, trivia.token());
          }
        }
      }
      previousTokenLine = token.line();
    });
  }

  private void addQuickFix(IssueWithQuickFix issue, Token commentToken) {
    String line = lines.get(commentToken.column());
    int column = 0;
    Matcher matcher = NON_WHITESPACE_PATTERN.matcher(line);
    if (matcher.matches()) {
      column = matcher.start();
    }
    String indent = " ".repeat(column);
    PythonTextEdit insert = new PythonTextEdit(indent + commentToken + "\n", commentToken.column(), 0, commentToken.column(), 0);
    PythonTextEdit remove = PythonTextEdit.removeRange(commentToken.line(), line.indexOf(commentToken.value()), commentToken.line(), line.length());

    PythonQuickFix fix = PythonQuickFix.newQuickFix(MESSAGE)
      .addTextEdit(List.of(insert, remove))
      .build();
    issue.addQuickFix(fix);
  }
}

