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
package org.sonar.python.checks.utils;

import java.util.ArrayList;
import java.util.List;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

/**
 * Shared helpers for single-invocation assertion checks (e.g. S5778, S9088):
 * collecting possibly-raising/warning calls inside {@code with} / lambda bodies
 * and reporting secondary locations on each unsafe invocation.
 */
public final class SingleInvocationUtils {

  /**
   * Calls that are almost never the exception/warning under test when nested inside
   * {@code pytest.raises} / {@code assertRaises} / {@code pytest.warns}.
   * Treating them as safe avoids noisy FPs on common setup helpers.
   */
  private static final TypeMatcher ALWAYS_SAFE_CALL_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("builtins.str"),
    TypeMatchers.isType("builtins.bytes"),
    TypeMatchers.isType("builtins.bytearray"),
    TypeMatchers.isType("builtins.repr"),
    TypeMatchers.isType("builtins.ascii"),
    TypeMatchers.isType("builtins.format"),
    TypeMatchers.isType("builtins.bool"),
    TypeMatchers.isType("builtins.int"),
    TypeMatchers.isType("builtins.float"),
    TypeMatchers.isType("builtins.complex"),
    TypeMatchers.isType("builtins.memoryview"),
    TypeMatchers.isType("builtins.list"),
    TypeMatchers.isType("builtins.tuple"),
    TypeMatchers.isType("builtins.dict"),
    TypeMatchers.isType("builtins.print"),
    TypeMatchers.isType("builtins.len"),
    TypeMatchers.isType("builtins.abs"),
    TypeMatchers.isType("builtins.round"),
    TypeMatchers.isType("builtins.id"),
    TypeMatchers.isType("builtins.hash"),
    TypeMatchers.isType("builtins.hex"),
    TypeMatchers.isType("builtins.oct"),
    TypeMatchers.isType("builtins.bin"),
    TypeMatchers.isType("builtins.ord"),
    TypeMatchers.isType("builtins.chr"),
    TypeMatchers.isType("builtins.range"),
    TypeMatchers.isType("builtins.enumerate"),
    TypeMatchers.isType("builtins.zip"),
    TypeMatchers.isType("builtins.reversed"),
    TypeMatchers.isType("builtins.sorted"),
    TypeMatchers.isType("builtins.slice"),
    TypeMatchers.isType("builtins.callable"),
    TypeMatchers.isType("pathlib.Path"),
    TypeMatchers.isType("pathlib.PurePath"),
    TypeMatchers.isType("pathlib.PosixPath"),
    TypeMatchers.isType("pathlib.WindowsPath"),
    TypeMatchers.isType("pathlib.PurePosixPath"),
    TypeMatchers.isType("pathlib.PureWindowsPath"),
    TypeMatchers.isType("uuid.UUID"),
    TypeMatchers.isType("uuid.uuid1"),
    TypeMatchers.isType("uuid.uuid3"),
    TypeMatchers.isType("uuid.uuid4"),
    TypeMatchers.isType("uuid.uuid5"),
    TypeMatchers.isType("uuid.uuid6"),
    TypeMatchers.isType("uuid.uuid7"),
    TypeMatchers.isType("uuid.uuid8"),
    TypeMatchers.isType("copy.copy"),
    TypeMatchers.isType("copy.deepcopy"),
    // NumPy/SciPy lack stubs → UnresolvedImportType; withFQNPrefix covers factories/helpers
    // (zeros, random, copy, legendre, …) used as nested setup in exception/warning assertions.
    TypeMatchers.withFQNPrefix("numpy."),
    TypeMatchers.withFQNPrefix("scipy."));

  /**
   * Constructors that are safe only without arguments (with args they commonly raise TypeError
   * or can behave unexpectedly).
   */
  private static final TypeMatcher EMPTY_ARGS_SAFE_CALL_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("builtins.set"),
    TypeMatchers.isType("builtins.frozenset"),
    TypeMatchers.isType("builtins.object"));

  private SingleInvocationUtils() {
  }

  public static List<CallExpression> unsafeInvocations(Tree tree, SubscriptionContext ctx) {
    var visitor = new InvocationCollector(ctx);
    tree.accept(visitor);
    return visitor.invocations.stream()
      .sorted(TreeUtils.getTreeByPositionComparator())
      .toList();
  }

  public static void reportIfMultipleInvocations(PythonCheck.PreciseIssue issue, List<CallExpression> invocations,
    String secondaryMessage) {
    invocations.forEach(invocation -> issue.secondary(invocationLocation(invocation, secondaryMessage)));
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

  private static class InvocationCollector extends BaseTreeVisitor {
    private final SubscriptionContext ctx;
    private final List<CallExpression> invocations = new ArrayList<>();

    private InvocationCollector(SubscriptionContext ctx) {
      this.ctx = ctx;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (!isSafeCall(callExpression)) {
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

    private boolean isSafeCall(CallExpression callExpression) {
      if (ALWAYS_SAFE_CALL_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
        return true;
      }
      return EMPTY_ARGS_SAFE_CALL_MATCHER.isTrueFor(callExpression.callee(), ctx)
        && callExpression.arguments().isEmpty();
    }
  }
}
