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
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9002")
public class UseTmpPathInsteadOfFactoryCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use \"tmp_path\" instead of \"tmp_path_factory\" in function-scoped tests.";
  private static final String TMP_PATH_FACTORY = "tmp_path_factory";
  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN(UnittestUtils.PYTEST_FIXTURE_DECORATOR_FQN);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, UseTmpPathInsteadOfFactoryCheck::checkFunction);
  }

  private static void checkFunction(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (!UnittestUtils.isPytestStyleTestFunction(functionDef, ctx.pythonFile().fileName())) {
      return;
    }
    if (isPytestFixture(functionDef, ctx)) {
      return;
    }

    for (Parameter parameter : TreeUtils.nonTupleParameters(functionDef)) {
      Name name = parameter.name();
      if (name != null && TMP_PATH_FACTORY.equals(name.name())) {
        ctx.addIssue(name, MESSAGE);
      }
    }
  }

  private static boolean isPytestFixture(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .anyMatch(decorator -> PYTEST_FIXTURE_MATCHER.isTrueFor(decoratorExpression(decorator), ctx));
  }

  private static Expression decoratorExpression(Decorator decorator) {
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
