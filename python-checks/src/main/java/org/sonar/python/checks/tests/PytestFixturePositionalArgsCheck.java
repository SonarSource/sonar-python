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
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S9116")
public class PytestFixturePositionalArgsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Pass fixture options as keyword arguments.";

  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN("pytest.fixture");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, PytestFixturePositionalArgsCheck::checkDecorator);
  }

  private static void checkDecorator(SubscriptionContext ctx) {
    Decorator decorator = (Decorator) ctx.syntaxNode();
    if (!(decorator.expression() instanceof CallExpression callExpression)) {
      return;
    }
    if (!PYTEST_FIXTURE_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    for (Argument argument : callExpression.arguments()) {
      if (isPositionalArgument(argument)) {
        ctx.addIssue(argument, MESSAGE);
        return;
      }
    }
  }

  private static boolean isPositionalArgument(Argument argument) {
    if (argument instanceof RegularArgument regularArgument) {
      return regularArgument.keywordArgument() == null;
    }
    if (argument instanceof UnpackingExpression unpackingExpression) {
      return "*".equals(unpackingExpression.starToken().value());
    }
    return false;
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
