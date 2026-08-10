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
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.UnittestUtils;

@Rule(key = "S9100")
public class PytestUselessYieldFixtureCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_USE_RETURN = "No teardown in this fixture. Use \"return\" instead of \"yield\".";
  private static final String MESSAGE_REMOVE_YIELD = "Remove this useless \"yield\".";
  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN(UnittestUtils.PYTEST_FIXTURE_DECORATOR_FQN);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, PytestUselessYieldFixtureCheck::checkFunctionDef);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (!isPytestFixture(functionDef, ctx) || CheckUtils.isAbstract(functionDef)) {
      return;
    }

    List<Statement> statements = functionDef.body().statements();
    if (statements.isEmpty()) {
      return;
    }

    Statement last = statements.get(statements.size() - 1);
    if (!(last instanceof YieldStatement yieldStatement)) {
      return;
    }
    // "yield from" delegates teardown to another generator (Ruff PT022 ignores it).
    if (yieldStatement.yieldExpression().fromKeyword() != null) {
      return;
    }

    // Multiple yields are covered by S8994; telling the user to "return" / remove the last
    // yield would be wrong advice when an earlier yield already made this a generator fixture.
    // Same guard as Ruff PT022 (exactly one yield in the fixture body).
    YieldCounter yieldCounter = new YieldCounter();
    functionDef.body().accept(yieldCounter);
    if (yieldCounter.count == 1) {
      boolean bareYield = yieldStatement.yieldExpression().expressions().isEmpty();
      ctx.addIssue(yieldStatement, bareYield ? MESSAGE_REMOVE_YIELD : MESSAGE_USE_RETURN);
    }
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

  private static final class YieldCounter extends BaseTreeVisitor {
    private int count;

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Skip nested functions
    }

    @Override
    public void visitLambda(LambdaExpression lambdaExpression) {
      // Skip lambdas
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpression) {
      count++;
    }
  }
}
