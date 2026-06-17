/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.plugins.python.api.nosonar.NoSonarInfoParser;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;

@Rule(key = "S1309")
public class NoQaCommentCheck extends PythonSubscriptionCheck {

  private static final String NOQA_MESSAGE = "Is 'noqa' used to exclude false-positive or to hide real quality flaw?";
  private static final String NOSEC_MESSAGE = "Is 'nosec' used to exclude false-positive or to hide real security issue?";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
      Token token = (Token) ctx.syntaxNode();
      for (Trivia trivia : token.trivia()) {
        for (String comment : NoSonarInfoParser.splitInlineComments(trivia.token().value())) {
          String message = null;
          if (NoSonarInfoParser.isValidNoQa(comment)) {
            message = NOQA_MESSAGE;
          } else if (NoSonarInfoParser.isValidNoSec(comment)) {
            message = NOSEC_MESSAGE;
          }
          if (message != null) {
            ctx.addIssue(trivia.token(), message);
            break;
          }
        }
      }
    });
  }
}
