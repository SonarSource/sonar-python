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

