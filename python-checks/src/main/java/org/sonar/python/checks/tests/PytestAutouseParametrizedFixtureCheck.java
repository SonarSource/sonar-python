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
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8992")
public class PytestAutouseParametrizedFixtureCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove the \"params\" argument or set \"autouse\" to False.";
  private static final String AUTOUSE_ARGUMENT = "autouse";
  private static final String PARAMS_ARGUMENT = "params";

  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN("pytest.fixture");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, PytestAutouseParametrizedFixtureCheck::checkDecorator);
  }

  private static void checkDecorator(SubscriptionContext ctx) {
    Decorator decorator = (Decorator) ctx.syntaxNode();
    Expression decoratorFunctionExpression = getDecoratorFunctionExpression(decorator);
    if (!PYTEST_FIXTURE_MATCHER.isTrueFor(decoratorFunctionExpression, ctx)) {
      return;
    }

    ArgList argList = decorator.arguments();
    if (argList == null) {
      return;
    }

    RegularArgument autouseArgument = TreeUtils.argumentByKeyword(AUTOUSE_ARGUMENT, argList.arguments());
    if (autouseArgument == null || !Expressions.isTruthy(autouseArgument.expression())) {
      return;
    }

    RegularArgument paramsArgument = TreeUtils.argumentByKeyword(PARAMS_ARGUMENT, argList.arguments());
    if (paramsArgument == null || Expressions.isFalsy(paramsArgument.expression())) {
      return;
    }

    var keywordArgument = paramsArgument.keywordArgument();
    if (keywordArgument != null) {
      ctx.addIssue(keywordArgument, MESSAGE);
    }
  }

  private static Expression getDecoratorFunctionExpression(Decorator decorator) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpression) {
      return callExpression.callee();
    }
    return expression;
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
