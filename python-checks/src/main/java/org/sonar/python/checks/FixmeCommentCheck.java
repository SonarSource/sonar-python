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
import org.sonar.python.PythonCheck;

@Rule(key = FixmeCommentCheck.CHECK_KEY)
public class FixmeCommentCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1134";

  private static final String FIXME_COMMENT_PATTERN = "^#[ ]*fixme.*";
  private static final String MESSAGE = "Take the required action to fix the issue indicated by this \"FIXME\" comment.";

  private Pattern pattern;

  @Override
  public void visitFile(AstNode astNode) {
    pattern = Pattern.compile(FIXME_COMMENT_PATTERN, Pattern.CASE_INSENSITIVE);
  }

  @Override
  public void visitToken(Token token) {
    for (Trivia trivia : token.getTrivia()) {
      String comment = trivia.getToken().getValue();
      if (pattern.matcher(comment).matches()) {
        addIssue(trivia.getToken(), MESSAGE);
      }
    }
  }
}

