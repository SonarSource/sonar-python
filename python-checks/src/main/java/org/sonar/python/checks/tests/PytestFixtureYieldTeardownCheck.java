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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.semantic.v2.typeshed.TypeShedConstants;

@Rule(key = "S9080")
public class PytestFixtureYieldTeardownCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this \"request.addfinalizer()\" call with a yield-based teardown.";
  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN(UnittestUtils.PYTEST_FIXTURE_DECORATOR_FQN);
  private static final TypeMatcher ADD_FINALIZER_MATCHER = TypeMatchers.isType(TypeShedConstants.PYTEST_FIXTURE_REQUEST_ADD_FINALIZER_FQN);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, PytestFixtureYieldTeardownCheck::checkFunctionDef);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (!isPytestFixture(functionDef, ctx)) {
      return;
    }

    var visitor = new FinalizerVisitor(ctx);
    functionDef.body().accept(visitor);
    if (visitor.hasYieldWithValue) {
      return;
    }
    visitor.finalizerCallees.forEach(callee -> ctx.addIssue(callee, MESSAGE));
  }

  private static boolean isPytestFixture(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .anyMatch(decorator -> PYTEST_FIXTURE_MATCHER.isTrueFor(decoratorFunctionExpression(decorator), ctx));
  }

  private static Expression decoratorFunctionExpression(Decorator decorator) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpression) {
      return callExpression.callee();
    }
    return expression;
  }

  /**
   * Collects the {@code request.addfinalizer} calls registered by the fixture itself. Nested functions and lambdas are not
   * visited: registering a finalizer there is the factory-as-fixture pattern, which cannot be expressed with a yield.
   */
  private static class FinalizerVisitor extends BaseTreeVisitor {

    private final SubscriptionContext ctx;
    private final List<Expression> finalizerCallees = new ArrayList<>();
    private boolean hasYieldWithValue = false;

    FinalizerVisitor(SubscriptionContext ctx) {
      this.ctx = ctx;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Expression callee = callExpression.callee();
      if (ADD_FINALIZER_MATCHER.isTrueFor(callee, ctx)) {
        finalizerCallees.add(callee);
      }
      super.visitCallExpression(callExpression);
    }

    @Override
    public void visitYieldStatement(YieldStatement yieldStatement) {
      if (yieldsValue(yieldStatement.yieldExpression())) {
        hasYieldWithValue = true;
      }
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpression) {
      if (yieldsValue(yieldExpression)) {
        hasYieldWithValue = true;
      }
    }

    private static boolean yieldsValue(YieldExpression yieldExpression) {
      return !yieldExpression.expressions().isEmpty();
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // nested functions belong to the factory-as-fixture pattern
    }

    @Override
    public void visitLambda(LambdaExpression lambdaExpression) {
      // see visitFunctionDef
    }
  }
}
