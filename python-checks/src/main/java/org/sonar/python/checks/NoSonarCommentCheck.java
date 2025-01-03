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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;

import static org.sonar.python.metrics.FileLinesVisitor.containsNoSonarComment;


/*
 * When updating this rule through the rule-api the sqKey present in the NoSonar.json file
 * should be kept to `NoSonar` instead of `S1291`
 */
@Rule(key = "NoSonar")
public class NoSonarCommentCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Is #NOSONAR used to exclude false-positive or to hide real quality flaw?";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
      Token token = (Token) ctx.syntaxNode();
      for (Trivia trivia : token.trivia()) {
        if (containsNoSonarComment(trivia)) {
          ctx.addIssue(trivia.token(), MESSAGE);
        }
      }
    });
  }
}

