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
package org.sonar.python.checks;

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.matchers.InternalTypeMatchers;
import org.sonar.python.types.v2.matchers.IsStubFunctionPredicate;
import org.sonar.python.types.v2.matchers.IsTypePredicate;
import org.sonar.python.types.v2.matchers.TypePredicate;
import org.sonar.python.types.v2.matchers.TypePredicateContext;
import org.sonar.python.types.v2.matchers.TypePredicateUtils;

@Rule(key = "S9408")
public class NonContextManagerCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Change this expression to a context manager (implementing \"__enter__\" and \"__exit__\").";
  private static final String ASYNC_MESSAGE = "Change this expression to an asynchronous context manager (implementing \"__aenter__\" and \"__aexit__\").";
  private static final String USE_WITH_MESSAGE = "Use \"with\" for this context manager created by \"@contextmanager\".";
  private static final String USE_ASYNC_WITH_MESSAGE = "Use \"async with\" for this asynchronous context manager created by \"@asynccontextmanager\".";

  private static final TypeMatcher SYNC_PROTOCOL = TypeMatchers.all(TypeMatchers.hasMember("__enter__"), TypeMatchers.hasMember("__exit__"));
  private static final TypeMatcher ASYNC_PROTOCOL = TypeMatchers.all(TypeMatchers.hasMember("__aenter__"), TypeMatchers.hasMember("__aexit__"));
  private static final TypeMatcher IS_CONTEXT_MANAGER = TypeMatchers.any(SYNC_PROTOCOL, InternalTypeMatchers.isAnyTypeInUnionSatisfying(SYNC_PROTOCOL));
  private static final TypeMatcher IS_ASYNC_CONTEXT_MANAGER = TypeMatchers.any(ASYNC_PROTOCOL, InternalTypeMatchers.isAnyTypeInUnionSatisfying(ASYNC_PROTOCOL));
  private static final TypeMatcher IS_GENERATOR = TypeMatchers.any(TypeMatchers.isObjectInstanceOf("typing.Generator"), TypeMatchers.isObjectInstanceOf("typing.Iterator"));
  private static final TypeMatcher IS_ASYNC_GENERATOR = TypeMatchers.any(
          TypeMatchers.isObjectInstanceOf("typing.Coroutine"),
          TypeMatchers.isObjectInstanceOf("typing.AsyncGenerator"),
          TypeMatchers.isObjectInstanceOf("typing.AsyncIterator"));

  private static final TypePredicate STUB_FUNCTION = new IsStubFunctionPredicate();
  private static final TypePredicate SYNC_CONTEXT_MANAGER_DECORATOR = new IsTypePredicate("contextlib.contextmanager");
  private static final TypePredicate ASYNC_CONTEXT_MANAGER_DECORATOR = new IsTypePredicate("contextlib.asynccontextmanager");

  // Builtin decorators that keep the wrapped function's return type and therefore cannot turn a generator into a context manager.
  private static final Set<String> SIGNATURE_PRESERVING_DECORATORS = Set.of(
    "staticmethod", "classmethod", "abstractmethod", "property", "overload", "cache", "final", "override");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_ITEM, NonContextManagerCheck::checkWithItem);
  }

  /**
   * <pre>{@code
   * from contextlib import contextmanager, asynccontextmanager
   *
   * @contextmanager
   * def sync_cm(): yield 1
   * @asynccontextmanager
   * async def async_cm(): yield 1
   *
   * async def use():
   *     with sync_cm(): pass         # valid
   *     async with sync_cm(): pass   # S9408
   *     async with async_cm(): pass  # valid
   *     with async_cm(): pass        # S9408
   * }</pre>
   */
  private static void checkWithItem(SubscriptionContext ctx) {
    WithItem item = (WithItem) ctx.syntaxNode();
    Expression contextManager = item.test();
    boolean isAsync = ((WithStatement) item.parent()).isAsync();

    if (contextManager.is(Tree.Kind.GENERATOR_EXPR)) {
      ctx.addIssue(contextManager, isAsync ? ASYNC_MESSAGE : MESSAGE);
      return;
    }

    checkContextManager(ctx, item, contextManager, isAsync);
  }

  private static void checkContextManager(SubscriptionContext ctx, WithItem item, Expression contextManager, boolean isAsync) {
    TypeMatcher matcher = isAsync ? IS_ASYNC_CONTEXT_MANAGER : IS_CONTEXT_MANAGER;
    if (!isDefinitelyNot(matcher, contextManager, ctx)) {
      return;
    }

    boolean hasMismatchedDecorator = hasMismatchedContextManagerDecorator(contextManager, isAsync, ctx);
    if (isUnmodeledContextManager(contextManager, isAsync, hasMismatchedDecorator, ctx)) {
      return;
    }
    if (!isAsync && isReportedByAsyncWithCheck(item, contextManager, ctx)) {
      return;
    }

    ctx.addIssue(contextManager, contextManagerMessage(isAsync, hasMismatchedDecorator));
  }

  private static String contextManagerMessage(boolean isAsync, boolean hasMismatchedDecorator) {
    if (hasMismatchedDecorator) {
      return isAsync ? USE_WITH_MESSAGE : USE_ASYNC_WITH_MESSAGE;
    }
    return isAsync ? ASYNC_MESSAGE : MESSAGE;
  }

  private static boolean isDefinitelyNot(TypeMatcher matcher, Expression expression, SubscriptionContext ctx) {
    return matcher.evaluateFor(expression, ctx) == TriBool.FALSE;
  }

  // Decorator effects are not modeled by inference. A known context manager decorator only supports its matching statement.
  private static boolean isUnmodeledContextManager(Expression expression, boolean isAsync, boolean hasMismatchedDecorator, SubscriptionContext ctx) {
    if (expression instanceof CallExpression call && call.callee().typeV2() instanceof FunctionType functionType) {
      return !hasMismatchedDecorator && hasUnknownDecorator(functionType);
    }
    TypeMatcher generatorMatcher = isAsync ? IS_ASYNC_GENERATOR : IS_GENERATOR;
    return generatorMatcher.evaluateFor(expression, ctx) == TriBool.TRUE;
  }

  private static boolean hasMismatchedContextManagerDecorator(Expression expression, boolean isAsync, SubscriptionContext ctx) {
    return expression instanceof CallExpression call
      && call.callee().typeV2() instanceof FunctionType functionType
      && hasOnlyMismatchedContextManagerDecorator(functionType, isAsync, ctx);
  }

  private static boolean hasOnlyMismatchedContextManagerDecorator(FunctionType functionType, boolean isAsync, SubscriptionContext ctx) {
    if (functionType.decorators().size() != 1) {
      return false;
    }
    PythonType decorator = functionType.decorators().get(0).type();
    var predicateContext = TypePredicateContext.of(ctx.typeTable());
    // Library identity may be overloaded; project-local contextlib replacements remain unknown decorators.
    TypePredicate incompatibleDecorator = isAsync ? SYNC_CONTEXT_MANAGER_DECORATOR : ASYNC_CONTEXT_MANAGER_DECORATOR;
    return incompatibleDecorator.check(decorator, predicateContext).isTrue()
      && TypePredicateUtils.evaluate(STUB_FUNCTION, decorator, predicateContext).isTrue();
  }

  private static boolean hasUnknownDecorator(FunctionType functionType) {
    return functionType.decorators().stream()
      .map(TypeWrapper::type)
      .map(PythonType::name)
      .anyMatch(name -> !SIGNATURE_PRESERVING_DECORATORS.contains(name));
  }

  // S7515 reports a known async context manager used with "with" inside an async function.
  private static boolean isReportedByAsyncWithCheck(WithItem withItem, Expression contextManager, SubscriptionContext ctx) {
    return TreeUtils.asyncTokenOfEnclosingFunction(withItem).isPresent()
      && IS_ASYNC_CONTEXT_MANAGER.evaluateFor(contextManager, ctx) == TriBool.TRUE;
  }
}
