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
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.matchers.InternalTypeMatchers;

@Rule(key = "S9408")
public class NonContextManagerCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Change this expression to a context manager (implementing \"__enter__\" and \"__exit__\").";
  private static final String ASYNC_MESSAGE = "Change this expression to an asynchronous context manager (implementing \"__aenter__\" and \"__aexit__\").";

  // Matched by simple name so back-ported copies are covered too.
  private static final String CONTEXT_MANAGER_DECORATOR = "contextmanager";
  private static final String ASYNC_CONTEXT_MANAGER_DECORATOR = "asynccontextmanager";
  private static final TypeMatcher SYNC_PROTOCOL = TypeMatchers.all(TypeMatchers.hasMember("__enter__"), TypeMatchers.hasMember("__exit__"));
  private static final TypeMatcher ASYNC_PROTOCOL = TypeMatchers.all(TypeMatchers.hasMember("__aenter__"), TypeMatchers.hasMember("__aexit__"));
  private static final TypeMatcher IS_CONTEXT_MANAGER = TypeMatchers.any(SYNC_PROTOCOL, InternalTypeMatchers.isAnyTypeInUnionSatisfying(SYNC_PROTOCOL));
  private static final TypeMatcher IS_ASYNC_CONTEXT_MANAGER = TypeMatchers.any(ASYNC_PROTOCOL, InternalTypeMatchers.isAnyTypeInUnionSatisfying(ASYNC_PROTOCOL));
  private static final TypeMatcher IS_GENERATOR = TypeMatchers.any(TypeMatchers.isObjectInstanceOf("typing.Generator"), TypeMatchers.isObjectInstanceOf("typing.Iterator"));
  private static final TypeMatcher IS_ASYNC_GENERATOR = TypeMatchers.any(
          TypeMatchers.isObjectInstanceOf("typing.Coroutine"),
          TypeMatchers.isObjectInstanceOf("typing.AsyncGenerator"),
          TypeMatchers.isObjectInstanceOf("typing.AsyncIterator"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_ITEM, NonContextManagerCheck::checkWithItem);
  }

  private static void checkWithItem(SubscriptionContext ctx) {
    WithItem item = (WithItem) ctx.syntaxNode();
    Expression contextManager = item.test();
    boolean isAsync = ((WithStatement) item.parent()).isAsync();

    if (contextManager.is(Tree.Kind.GENERATOR_EXPR)) {
      ctx.addIssue(contextManager, isAsync ? ASYNC_MESSAGE : MESSAGE);
      return;
    }

    if (isAsync) {
      if (isDefinitelyNot(IS_ASYNC_CONTEXT_MANAGER, contextManager, ctx)
        && !isUnmodeledContextManager(contextManager, ASYNC_CONTEXT_MANAGER_DECORATOR, IS_ASYNC_GENERATOR, ctx)) {
        ctx.addIssue(contextManager, ASYNC_MESSAGE);
      }
    } else if (isDefinitelyNot(IS_CONTEXT_MANAGER, contextManager, ctx)
      && !isUnmodeledContextManager(contextManager, CONTEXT_MANAGER_DECORATOR, IS_GENERATOR, ctx)
      && !isReportedByAsyncWithCheck(item, contextManager, ctx)) {
      ctx.addIssue(contextManager, MESSAGE);
    }
  }

  private static boolean isDefinitelyNot(TypeMatcher matcher, Expression expression, SubscriptionContext ctx) {
    return matcher.evaluateFor(expression, ctx) == TriBool.FALSE;
  }

  // @contextmanager is not modeled by inference: detect it via the decorator on direct calls, else trust the generator type.
  private static boolean isUnmodeledContextManager(Expression expression, String decoratorName, TypeMatcher generatorMatcher, SubscriptionContext ctx) {
    if (expression instanceof CallExpression call && call.callee().typeV2() instanceof FunctionType functionType) {
      return isDecoratedWith(functionType, decoratorName);
    }
    return generatorMatcher.evaluateFor(expression, ctx) == TriBool.TRUE;
  }

  private static boolean isDecoratedWith(FunctionType functionType, String decoratorName) {
    return functionType.decorators().stream()
      .map(TypeWrapper::type)
      .anyMatch(type -> type instanceof FunctionType decorator && decoratorName.equals(decorator.name()));
  }

  // An async context manager used with a plain "with" inside an async function is reported by S7515, so skip it here.
  private static boolean isReportedByAsyncWithCheck(WithItem withItem, Expression contextManager, SubscriptionContext ctx) {
    return TreeUtils.asyncTokenOfEnclosingFunction(withItem).isPresent()
      && IS_ASYNC_CONTEXT_MANAGER.evaluateFor(contextManager, ctx) == TriBool.TRUE;
  }
}
