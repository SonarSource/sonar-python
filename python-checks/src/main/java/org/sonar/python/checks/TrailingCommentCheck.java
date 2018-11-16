/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheck;

@Rule(key = TrailingCommentCheck.CHECK_KEY)
public class TrailingCommentCheck extends PythonCheck {

  public static final String CHECK_KEY = "S139";
  private static final String DEFAULT_LEGAL_COMMENT_PATTERN = "^#\\s*+[^\\s]++$";
  private static final String MESSAGE = "Move this trailing comment on the previous empty line.";

  @RuleProperty(
    key = "legalTrailingCommentPattern",
    defaultValue = DEFAULT_LEGAL_COMMENT_PATTERN)
  public String legalCommentPattern = DEFAULT_LEGAL_COMMENT_PATTERN;

  private Pattern pattern;
  private int previousTokenLine;

  @Override
  public void visitFile(AstNode astNode) {
    previousTokenLine = -1;
    pattern = Pattern.compile(legalCommentPattern);
  }

  @Override
  public void visitToken(Token token) {
    for (Trivia trivia : token.getTrivia()) {
      if (trivia.getToken().getLine() == previousTokenLine) {
        String comment = trivia.getToken().getValue();
        if (!pattern.matcher(comment).matches()) {
          addIssue(trivia.getToken(), MESSAGE);
        }
      }
    }
    previousTokenLine = token.getLine();
  }
}

