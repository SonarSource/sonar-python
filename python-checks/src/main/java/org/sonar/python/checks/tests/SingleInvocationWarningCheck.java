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
package org.sonar.python.checks.tests;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.python.checks.utils.SingleInvocationUtils;
import org.sonar.python.checks.utils.UnittestUtils;

@Rule(key = "S9088")
public class SingleInvocationWarningCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Refactor this warning test to have only one invocation possibly emitting a warning.";
  private static final String SECONDARY_MESSAGE = "Invocation possibly emitting a warning.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, ctx -> checkWithStatement(ctx, (WithStatement) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkWithStatement(SubscriptionContext ctx, WithStatement withStatement) {
    boolean isWarnsAssertion = withStatement.withItems().stream()
      .map(WithItem::test)
      .filter(CallExpression.class::isInstance)
      .map(CallExpression.class::cast)
      .anyMatch(callExpression -> UnittestUtils.isPytestWarns(callExpression, ctx));

    if (!isWarnsAssertion) {
      return;
    }

    var invocations = SingleInvocationUtils.unsafeInvocations(withStatement.statements(), ctx);
    if (invocations.size() > 1) {
      SingleInvocationUtils.reportIfMultipleInvocations(
        ctx.addIssue(withStatement.withKeyword(), withStatement.colon(), MESSAGE),
        invocations,
        SECONDARY_MESSAGE);
    }
  }
}
