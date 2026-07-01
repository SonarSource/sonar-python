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
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.python.checks.utils.UnittestUtils;

@Rule(key = "S5778")
public class SingleInvocationRuntimeExceptionCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Refactor this exception test to have only one invocation possibly throwing an exception.";
  private static final String SECONDARY_MESSAGE = "Invocation possibly throwing an exception.";
  private static final String PYTEST_RAISES = "pytest.raises";
  private static final List<String> SAFE_BUILTINS = List.of(
    "list",
    "set",
    "dict",
    "tuple",
    "frozenset",
    "str",
    "bytes",
    "bytearray",
    "object");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, ctx -> checkWithStatement(ctx, (WithStatement) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> checkDirectRaiseCall(ctx, (CallExpression) ctx.syntaxNode()));
  }

  private static void checkWithStatement(SubscriptionContext ctx, WithStatement withStatement) {
    boolean isRaiseAssertion = withStatement.withItems().stream()
      .map(WithItem::test)
      .filter(CallExpression.class::isInstance)
      .map(CallExpression.class::cast)
      .anyMatch(callExpression -> isPytestRaise(callExpression) || isUnittestRaise(callExpression));

    if (!isRaiseAssertion) {
      return;
    }

    var invocations = unsafeInvocations(withStatement.statements());
    if (invocations.size() > 1) {
      reportIfMultipleInvocations(ctx.addIssue(withStatement.withKeyword(), withStatement.colon(), MESSAGE), invocations);
    }
  }

  private static void checkDirectRaiseCall(SubscriptionContext ctx, CallExpression callExpression) {
    if (!isPytestRaise(callExpression) && !isUnittestRaise(callExpression)) {
      return;
    }

    findLambdaArgument(callExpression.arguments())
      .map(lambdaExpression -> unsafeInvocations(lambdaExpression.expression()))
      .filter(invocations -> invocations.size() > 1)
      .ifPresent(invocations -> reportIfMultipleInvocations(ctx.addIssue(callExpression, MESSAGE), invocations));
  }

  private static boolean isPytestRaise(CallExpression callExpression) {
    return Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(Objects::nonNull)
      .map(fqn -> fqn.contains(PYTEST_RAISES))
      .orElse(false);
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

  private static void reportIfMultipleInvocations(PythonCheck.PreciseIssue issue, List<CallExpression> invocations) {
    invocations.forEach(invocation -> issue.secondary(invocationLocation(invocation, SECONDARY_MESSAGE)));
  }

  private static IssueLocation invocationLocation(CallExpression invocation, String message) {
    if (invocation.callee() instanceof QualifiedExpression qualifiedExpression) {
      return IssueLocation.preciseLocation(qualifiedExpression.name().firstToken(), invocation.rightPar(), message);
    }
    if (invocation.callee() instanceof Name calleeName) {
      return IssueLocation.preciseLocation(calleeName.firstToken(), invocation.rightPar(), message);
    }
    return IssueLocation.preciseLocation(invocation, message);
  }

  private static List<CallExpression> unsafeInvocations(Tree tree) {
    var visitor = new InvocationCollector();
    tree.accept(visitor);
    return visitor.invocations.stream()
      .sorted(Comparator
        .comparingInt((CallExpression invocation) -> invocation.firstToken().line())
        .thenComparingInt(invocation -> invocation.firstToken().column())
        .thenComparingInt(invocation -> invocation.lastToken().line())
        .thenComparingInt(invocation -> invocation.lastToken().column()))
      .toList();
  }

  private static class InvocationCollector extends BaseTreeVisitor {
    private final List<CallExpression> invocations = new ArrayList<>();

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (!isSafeBuiltin(callExpression)) {
        invocations.add(callExpression);
      }
      super.visitCallExpression(callExpression);
    }

    @Override
    public void visitLambda(LambdaExpression lambdaExpression) {
      // Nested lambdas define deferred execution and should not contribute calls here.
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Nested function bodies are not executed when merely defined.
    }

    private static boolean isSafeBuiltin(CallExpression callExpression) {
      if (!(callExpression.callee() instanceof Name calleeName)) {
        return false;
      }

      String callee = calleeName.name();
      if (!SAFE_BUILTINS.contains(callee)) {
        return false;
      }

      return callExpression.arguments().isEmpty();
    }
  }
}
