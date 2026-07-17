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

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.semantic.v2.typeshed.TypeShedConstants;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8993")
public class FixtureParamDependenciesCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Declare this fixture as a test function parameter instead of using \"request.getfixturevalue()\" with a string literal.";
  private static final String CONFTEST_FILE_NAME = "conftest.py";
  private static final String PYTEST_HOOK_PREFIX = "pytest_";
  private static final TypeMatcher GET_FIXTURE_VALUE_MATCHER = TypeMatchers.isType(
    TypeShedConstants.PYTEST_FIXTURE_REQUEST_GET_FIXTURE_VALUE_FQN
  );
  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN(UnittestUtils.PYTEST_FIXTURE_DECORATOR_FQN);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, FixtureParamDependenciesCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (!GET_FIXTURE_VALUE_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    FunctionDef enclosingFunction = (FunctionDef) TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.FUNCDEF);
    if (enclosingFunction == null || !isCollectedTestFunction(ctx, enclosingFunction)) {
      return;
    }

    Expression fixtureNameExpression = fixtureNameArgument(callExpression);
    if (fixtureNameExpression == null || !fixtureNameExpression.is(Tree.Kind.STRING_LITERAL)) {
      return;
    }

    ctx.addIssue(fixtureNameExpression, MESSAGE);
  }

  private static boolean isCollectedTestFunction(SubscriptionContext ctx, FunctionDef functionDef) {
    if (isPytestFixture(functionDef, ctx)) {
      return false;
    }
    if (isPytestHook(functionDef)) {
      return false;
    }
    if (isConftestFile(ctx.pythonFile().fileName())) {
      return false;
    }
    return UnittestUtils.isPytestStyleTestFunction(functionDef, ctx.pythonFile().fileName());
  }

  private static boolean isPytestFixture(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .anyMatch(decorator -> PYTEST_FIXTURE_MATCHER.isTrueFor(getDecoratorFunctionExpression(decorator), ctx));
  }

  private static boolean isPytestHook(FunctionDef functionDef) {
    return functionDef.name().name().startsWith(PYTEST_HOOK_PREFIX);
  }

  private static boolean isConftestFile(String fileName) {
    return CONFTEST_FILE_NAME.equals(fileName);
  }

  private static Expression getDecoratorFunctionExpression(Decorator decorator) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpression) {
      return callExpression.callee();
    }
    return expression;
  }

  @Nullable
  private static Expression fixtureNameArgument(CallExpression callExpression) {
    RegularArgument argument = TreeUtils.nthArgumentOrKeyword(0, "argname", callExpression.arguments());
    return argument == null ? null : argument.expression();
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
