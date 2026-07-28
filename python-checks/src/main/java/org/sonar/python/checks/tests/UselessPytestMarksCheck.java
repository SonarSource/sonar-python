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
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S9074")
public class UselessPytestMarksCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_MARK_ON_FIXTURE = "Remove this mark; it has no effect on fixtures.";
  private static final String MESSAGE_EMPTY_USEFIXTURES = "Provide fixture names or remove this empty usefixtures decorator.";

  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN("pytest.fixture");
  private static final TypeMatcher PYTEST_MARK_MATCHER = TypeMatchers.isType("pytest.mark");
  private static final TypeMatcher PYTEST_USEFIXTURES_MATCHER = TypeMatchers.isType("pytest.mark.usefixtures");
  private static final TypeMatcher KNOWN_PYTEST_MARK_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("pytest.mark.skip"),
    TypeMatchers.isType("pytest.mark.xfail"),
    TypeMatchers.isType("pytest.mark.parametrize"),
    PYTEST_USEFIXTURES_MATCHER
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> checkFunction(ctx, (FunctionDef) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> checkClass(ctx, (ClassDef) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkFunction(SubscriptionContext ctx, FunctionDef functionDef) {
    boolean isFixture = hasPytestFixtureDecorator(functionDef.decorators(), ctx);
    for (Decorator decorator : functionDef.decorators()) {
      checkDecorator(ctx, decorator, isFixture);
    }
  }

  private static void checkClass(SubscriptionContext ctx, ClassDef classDef) {
    for (Decorator decorator : classDef.decorators()) {
      checkDecorator(ctx, decorator, false);
    }
  }

  private static void checkDecorator(SubscriptionContext ctx, Decorator decorator, boolean isFixture) {
    Expression markExpression = decoratorFunctionExpression(decorator);
    if (!isPytestMark(markExpression, ctx)) {
      return;
    }
    if (isFixture) {
      ctx.addIssue(decorator, MESSAGE_MARK_ON_FIXTURE);
      return;
    }
    if (isEmptyUsefixtures(decorator, markExpression, ctx)) {
      ctx.addIssue(decorator, MESSAGE_EMPTY_USEFIXTURES);
    }
  }

  private static boolean hasPytestFixtureDecorator(List<Decorator> decorators, SubscriptionContext ctx) {
    return decorators.stream()
      .anyMatch(decorator -> PYTEST_FIXTURE_MATCHER.isTrueFor(decoratorFunctionExpression(decorator), ctx));
  }

  private static boolean isPytestMark(Expression expression, SubscriptionContext ctx) {
    if (KNOWN_PYTEST_MARK_MATCHER.isTrueFor(expression, ctx)) {
      return true;
    }
    if (expression instanceof QualifiedExpression qualifiedExpression) {
      return PYTEST_MARK_MATCHER.isTrueFor(qualifiedExpression.qualifier(), ctx);
    }
    return false;
  }

  private static boolean isEmptyUsefixtures(Decorator decorator, Expression markExpression, SubscriptionContext ctx) {
    if (!PYTEST_USEFIXTURES_MATCHER.isTrueFor(markExpression, ctx)) {
      return false;
    }
    return hasEmptyCallArguments(decorator);
  }

  private static boolean hasEmptyCallArguments(Decorator decorator) {
    Expression expression = decorator.expression();
    if (!(expression instanceof CallExpression callExpression)) {
      return false;
    }
    ArgList argumentList = callExpression.argumentList();
    return argumentList == null || argumentList.arguments().isEmpty();
  }

  private static Expression decoratorFunctionExpression(Decorator decorator) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpression) {
      return callExpression.callee();
    }
    return expression;
  }
}
