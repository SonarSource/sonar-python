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
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
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
   * Calls whose sole purpose is to resolve or wrap a callable — the outer call is the actual
   * invocation under test, not this inner resolver.
   */
  private static final TypeMatcher CALLABLE_RESOLVER_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("builtins.getattr"),
    TypeMatchers.isType("operator.attrgetter"),
    TypeMatchers.isType("operator.methodcaller"),
    TypeMatchers.isType("functools.partial"));

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

  /**
   * Collects all call expressions within {@code tree} that are considered potentially unsafe
   * (i.e. could raise an exception or emit a warning under test).
   *
   * <p>A call is excluded from the result when any of the following conditions hold:
   * <ul>
   *   <li>It is a known-safe call (see {@code ALWAYS_SAFE_CALL_MATCHER} and
   *       {@code EMPTY_ARGS_SAFE_CALL_MATCHER}).</li>
   *   <li>It is an <em>imported</em> function used as an argument to another call —
   *       treated as setup data rather than the expression under test
   *       (e.g. {@code do_work(some_lib.factory())}).</li>
   *   <li>It is an empty-argument constructor call (uppercase callee name) used as the receiver of a
   *       method chain — only the chained method is the operation under test
   *       (e.g. {@code MyClass().some_method(args)}).</li>
   *   <li>It is a known callable-resolving or wrapping builtin ({@code getattr},
   *       {@code operator.attrgetter}, {@code operator.methodcaller}, {@code functools.partial})
   *       used as the direct callee of another call, or is the intermediate "apply" step in a
   *       3-call resolver chain (e.g. {@code attrgetter("attr")(obj)()} — all three calls are
   *       excluded and only the final {@code ()} invocation is counted).</li>
   * </ul>
   *
   * <p>Calls inside nested lambda or function definitions are not collected because they represent
   * deferred execution and do not run at the point of the assertion.
   *
   * @param tree the subtree to inspect (e.g. the body of a {@code with} statement or a lambda expression)
   * @param ctx  the subscription context used for type resolution
   * @return call expressions sorted by source position
   */
  public static List<CallExpression> unsafeInvocations(Tree tree, SubscriptionContext ctx) {
    var visitor = new InvocationCollector(ctx);
    tree.accept(visitor);
    return visitor.invocations.stream()
      .sorted(TreeUtils.getTreeByPositionComparator())
      .toList();
  }

  /**
   * Attaches each call expression in {@code invocations} as a secondary location on {@code issue}.
   * Callers are responsible for only invoking this when {@code invocations} contains more than one
   * entry; the primary issue location must be created by the caller before passing it here.
   *
   * @param issue            the primary issue to annotate with secondary locations
   * @param invocations      the unsafe call expressions collected by {@link #unsafeInvocations}
   * @param secondaryMessage the message to attach to each secondary location
   */
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
      if (!isKnownSafeCall(callExpression)
        && !isImportedCallUsedAsArgument(callExpression)
        && !isConstructorCallUsedAsChainReceiver(callExpression)
        && !isCallableResolverUsedAsCallee(callExpression)) {
        invocations.add(callExpression);
      }
      super.visitCallExpression(callExpression);
    }

    /**
     * Returns true when {@code callExpression} resolves to a type that is known never to be the
     * exception or warning under test:
     * <ul>
     *   <li>Calls always matched by {@code ALWAYS_SAFE_CALL_MATCHER} — common builtins, path
     *       utilities, uuid helpers, numpy/scipy factories, etc.</li>
     *   <li>Calls matched by {@code EMPTY_ARGS_SAFE_CALL_MATCHER} ({@code set}, {@code frozenset},
     *       {@code object}) <em>only when invoked with no arguments</em>; passing arguments to
     *       these constructors can itself raise.</li>
     * </ul>
     */
    private boolean isKnownSafeCall(CallExpression callExpression) {
      if (ALWAYS_SAFE_CALL_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
        return true;
      }
      return EMPTY_ARGS_SAFE_CALL_MATCHER.isTrueFor(callExpression.callee(), ctx)
              && callExpression.arguments().isEmpty();
    }

    /**
     * Returns true when {@code callExpression} is a <em>no-argument</em> constructor call used as
     * the qualifier of a dot expression — e.g. {@code MyClass().some_method(args)} or
     * {@code MyClass().attribute}. The no-argument constructor merely creates the receiver object
     * and is extremely unlikely to raise.
     *
     * <p>Constructor calls <em>with arguments</em> are intentionally not excluded.
     */
    private static boolean isConstructorCallUsedAsChainReceiver(CallExpression callExpression) {
      if (!callExpression.arguments().isEmpty() || !(callExpression.parent() instanceof QualifiedExpression)) {
        return false;
      }
      Expression callee = callExpression.callee();
      return callee instanceof Name name && Character.isUpperCase(name.name().charAt(0));
    }

    /**
     * Returns true when {@code callExpression} is the direct callee of another call expression
     * AND is a known callable-resolving builtin, or is the intermediate "apply" step in a
     * 3-call resolver chain (e.g. {@code operator.attrgetter("attr")(obj)()}).
     * These calls merely resolve the callable; the outer call is the actual invocation under test.
     */
    private boolean isCallableResolverUsedAsCallee(CallExpression callExpression) {
      if (!(callExpression.parent() instanceof CallExpression)) {
        return false;
      }
      // 2-call chain: resolver("attr")(args) — e.g. getattr, methodcaller, partial
      if (CALLABLE_RESOLVER_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
        return true;
      }
      // 3-call chain: resolver("attr")(obj)() — e.g. attrgetter; skip the intermediate apply step
      return callExpression.callee() instanceof CallExpression calleeCall
        && CALLABLE_RESOLVER_MATCHER.isTrueFor(calleeCall.callee(), ctx);
    }

    private static boolean isImportedCallUsedAsArgument(CallExpression callExpression) {
      Tree parent = callExpression.parent();
      if (!(parent instanceof RegularArgument)) {
        return false;
      }
      // RegularArgument appears not only in call argument lists but also in ClassDef base-class
      // lists and decorators. Restrict the "imported-as-argument" exclusion to calls only.
      Tree argList = parent.parent();
      if (argList == null || !(argList.parent() instanceof CallExpression)) {
        return false;
      }
      return isCalleeFromImport(callExpression);
    }

    private static boolean isCalleeFromImport(CallExpression callExpression) {
      Expression callee = callExpression.callee();
      while (callee instanceof QualifiedExpression qualifiedExpression) {
        callee = qualifiedExpression.qualifier();
      }
      return callee instanceof Name calleeName && isImportedName(calleeName);
    }

    private static boolean isImportedName(Name name) {
      SymbolV2 symbol = name.symbolV2();
      return symbol != null && symbol.usages().stream().anyMatch(u -> u.kind() == UsageV2.Kind.IMPORT);
    }

    @Override
    public void visitLambda(LambdaExpression lambdaExpression) {
      // Nested lambdas define deferred execution and should not contribute calls here.
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Nested function bodies are not executed when merely defined.
    }
  }
}
