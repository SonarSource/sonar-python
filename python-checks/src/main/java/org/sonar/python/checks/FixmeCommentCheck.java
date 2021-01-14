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
package org.sonar.python.checks;

import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;

@Rule(key = "S1134")
public class FixmeCommentCheck extends PythonSubscriptionCheck {

  private static final String FIXME_COMMENT_PATTERN = "^#[ ]*fixme.*";
  private static final String MESSAGE = "Take the required action to fix the issue indicated by this \"FIXME\" comment.";

  @Override
  public void initialize(Context context) {
    Pattern pattern = Pattern.compile(FIXME_COMMENT_PATTERN, Pattern.CASE_INSENSITIVE);
    context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
      Token token = (Token) ctx.syntaxNode();
      for (Trivia trivia : token.trivia()) {
        String comment = trivia.value();
        if (pattern.matcher(comment).matches()) {
          ctx.addIssue(trivia.token(), MESSAGE);
        }
      }
    });
  }
}

