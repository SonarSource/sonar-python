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
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;

@Rule(key = "S9073")
public class CompositeAssertionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Split this composite assertion into separate assertions.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> checkAssert(ctx, (AssertStatement) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkAssert(SubscriptionContext ctx, AssertStatement assertStatement) {
    if (!UnittestUtils.isPytestFileName(ctx.pythonFile().fileName())
      && !UnittestUtils.isSupportedTestFunction(ctx, assertStatement)) {
      return;
    }

    Expression condition = Expressions.removeParentheses(assertStatement.condition());

    if (condition.is(Tree.Kind.AND)) {
      ctx.addIssue(assertStatement, MESSAGE);
      return;
    }

    if (condition.is(Tree.Kind.NOT)) {
      Expression inner = Expressions.removeParentheses(((UnaryExpression) condition).expression());
      if (inner.is(Tree.Kind.OR)) {
        ctx.addIssue(assertStatement, MESSAGE);
      }
    }
  }
}
