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
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S9076")
public class DeprecatedYieldFixtureCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace deprecated pytest.yield_fixture with pytest.fixture.";

  private static final TypeMatcher YIELD_FIXTURE_MATCHER = TypeMatchers.isType("pytest.yield_fixture");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, ctx -> checkDecorator(ctx, (Decorator) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkDecorator(SubscriptionContext ctx, Decorator decorator) {
    Expression expression = decorator.expression();
    Expression matchedExpression = expression.is(Tree.Kind.CALL_EXPR) ? ((CallExpression) expression).callee() : expression;
    if (YIELD_FIXTURE_MATCHER.isTrueFor(matchedExpression, ctx)) {
      ctx.addIssue(expression, MESSAGE);
    }
  }
}
