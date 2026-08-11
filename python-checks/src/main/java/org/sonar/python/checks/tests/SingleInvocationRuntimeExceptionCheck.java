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
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.SingleInvocationUtils;
import org.sonar.python.checks.utils.UnittestUtils;

@Rule(key = "S5778")
public class SingleInvocationRuntimeExceptionCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Refactor this exception test to have only one invocation possibly throwing an exception.";
  private static final String SECONDARY_MESSAGE = "Invocation possibly throwing an exception.";

  private static final TypeMatcher PYTEST_RAISES_MATCHER = TypeMatchers.isType("pytest.raises");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, ctx -> checkWithStatement(ctx, (WithStatement) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> checkDirectRaiseCall(ctx, (CallExpression) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkWithStatement(SubscriptionContext ctx, WithStatement withStatement) {
    boolean isRaiseAssertion = withStatement.withItems().stream()
      .map(WithItem::test)
      .filter(CallExpression.class::isInstance)
      .map(CallExpression.class::cast)
      .anyMatch(callExpression -> isPytestRaise(callExpression, ctx) || isUnittestRaise(callExpression));

    if (!isRaiseAssertion) {
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

  private static void checkDirectRaiseCall(SubscriptionContext ctx, CallExpression callExpression) {
    if (!isPytestRaise(callExpression, ctx) && !isUnittestRaise(callExpression)) {
      return;
    }

    findLambdaArgument(callExpression.arguments())
      .map(lambdaExpression -> SingleInvocationUtils.unsafeInvocations(lambdaExpression.expression(), ctx))
      .filter(invocations -> invocations.size() > 1)
      .ifPresent(invocations -> SingleInvocationUtils.reportIfMultipleInvocations(
        ctx.addIssue(callExpression, MESSAGE),
        invocations,
        SECONDARY_MESSAGE));
  }

  private static boolean isPytestRaise(CallExpression callExpression, SubscriptionContext ctx) {
    return PYTEST_RAISES_MATCHER.isTrueFor(callExpression.callee(), ctx);
  }

  private static boolean isUnittestRaise(CallExpression callExpression) {
    if (!(callExpression.callee() instanceof QualifiedExpression qualifiedExpression)) {
      return false;
    }
    if (!(qualifiedExpression.qualifier() instanceof Name qualifier) || !"self".equals(qualifier.name())) {
      return false;
    }
    return UnittestUtils.isWithinUnittestTestCase(callExpression)
      && UnittestUtils.RAISE_METHODS.contains(qualifiedExpression.name().name());
  }

  private static Optional<LambdaExpression> findLambdaArgument(List<Argument> arguments) {
    return arguments.stream()
      .filter(RegularArgument.class::isInstance)
      .map(RegularArgument.class::cast)
      .map(RegularArgument::expression)
      .filter(LambdaExpression.class::isInstance)
      .map(LambdaExpression.class::cast)
      .findFirst();
  }
}
