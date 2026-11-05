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
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S9460")
public class SoleStarredGeneratorArgumentCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add parentheses around this starred generator argument.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.GENERATOR_EXPR, SoleStarredGeneratorArgumentCheck::checkGeneratorExpression);
  }

  private static void checkGeneratorExpression(SubscriptionContext ctx) {
    ComprehensionExpression generatorExpression = (ComprehensionExpression) ctx.syntaxNode();
    if (generatorExpression.parent() instanceof RegularArgument
      && generatorExpression.resultExpression().is(Tree.Kind.UNPACKING_EXPR)
      && !"(".equals(generatorExpression.firstToken().value())) {
      var issue = ctx.addIssue(generatorExpression, MESSAGE);
      var quickFix = PythonQuickFix.newQuickFix(MESSAGE)
        .addTextEdit(TextEditUtils.insertBefore(generatorExpression.firstToken(), "("))
        .addTextEdit(TextEditUtils.insertAfter(generatorExpression.lastToken(), ")"))
        .build();
      issue.addQuickFix(quickFix);
    }
  }
}
