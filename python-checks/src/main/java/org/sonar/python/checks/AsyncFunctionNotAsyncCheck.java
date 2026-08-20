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

import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AwaitExpression;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S7503")
public class AsyncFunctionNotAsyncCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use asynchronous features in this function or remove the `async` keyword.";
  private static final Set<String> AIO_CONSUMER_SUBSCRIBE_CALLBACK_KWARGS = Set.of("on_assign", "on_revoke", "on_lost");

  private static final TypeMatcher NOT_IMPLEMENTED_MATCHER = TypeMatchers.isType("builtins.NotImplemented");
  private static final TypeMatcher HTTPX_ASYNC_CLIENT_MATCHER = TypeMatchers.isOrExtendsType("httpx.AsyncClient");
  private static final TypeMatcher AIO_CONSUMER_SUBSCRIBE_MATCHER = TypeMatchers.isType("confluent_kafka.aio.AIOConsumer.subscribe");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, AsyncFunctionNotAsyncCheck::checkAsyncFunction);
  }

  private static void checkAsyncFunction(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    Token asyncKeyword = functionDef.asyncKeyword();
    if (asyncKeyword == null || isException(functionDef, ctx)) {
      return;
    }
    AsyncFeatureVisitor visitor = new AsyncFeatureVisitor();
    functionDef.body().accept(visitor);

    if (!visitor.hasAsyncFeature()) {
      ctx.addIssue(functionDef.name(), MESSAGE).secondary(asyncKeyword, "This function is async.");
    }
  }

  private static boolean isException(FunctionDef functionDef, SubscriptionContext ctx) {
    return CheckUtils.isAbstract(functionDef) ||
      isTrivialFunction(functionDef.body(), ctx) ||
      isDunderMethod(functionDef) ||
      !functionDef.decorators().isEmpty() ||
      mightBeOverridingMethod(functionDef) ||
      isExemptedCoroutineCallback(functionDef, ctx);
  }

  private static boolean isExemptedCoroutineCallback(FunctionDef functionDef, SubscriptionContext ctx) {
    SymbolV2 symbol = functionDef.name().symbolV2();
    if (symbol == null) {
      return false;
    }
    return symbol.usages().stream()
      .filter(usage -> usage.kind() == UsageV2.Kind.OTHER)
      .anyMatch(usage -> isCoroutineCallbackUsage(usage.tree(), ctx));
  }

  private static boolean isCoroutineCallbackUsage(Tree usageTree, SubscriptionContext ctx) {
    return isHttpxAsyncClientEventHookCallback(usageTree, ctx) || isAioConsumerSubscribeCallback(usageTree, ctx);
  }

  private static boolean isHttpxAsyncClientEventHookCallback(Tree usageTree, SubscriptionContext ctx) {
    CallExpression call = TreeUtils.firstAncestorOfClass(usageTree, CallExpression.class);
    if (call == null || !HTTPX_ASYNC_CLIENT_MATCHER.isTrueFor(call.callee(), ctx)) {
      return false;
    }
    RegularArgument eventHooksArg = TreeUtils.argumentByKeyword("event_hooks", call.arguments());
    if (eventHooksArg == null || !(eventHooksArg.expression() instanceof DictionaryLiteral dict)) {
      return false;
    }
    return dict.elements().stream()
      .filter(KeyValuePair.class::isInstance)
      .map(KeyValuePair.class::cast)
      .filter(kv -> isRequestOrResponseKey(kv.key()))
      .map(KeyValuePair::value)
      .filter(ListLiteral.class::isInstance)
      .map(ListLiteral.class::cast)
      .map(ListLiteral::elements)
      .map(ExpressionList::expressions)
      .flatMap(List::stream)
      .anyMatch(expr -> expr == usageTree);
  }

  private static boolean isRequestOrResponseKey(Expression key) {
    return key instanceof StringLiteral stringLiteral &&
      ("request".equals(stringLiteral.trimmedQuotesValue()) || "response".equals(stringLiteral.trimmedQuotesValue()));
  }

  private static boolean isAioConsumerSubscribeCallback(Tree usageTree, SubscriptionContext ctx) {
    RegularArgument arg = TreeUtils.firstAncestorOfClass(usageTree, RegularArgument.class);
    if (arg == null || arg.expression() != usageTree) {
      return false;
    }
    Name keyword = arg.keywordArgument();
    if (keyword == null || !AIO_CONSUMER_SUBSCRIBE_CALLBACK_KWARGS.contains(keyword.name())) {
      return false;
    }
    CallExpression call = TreeUtils.firstAncestorOfClass(arg, CallExpression.class);
    return call != null && AIO_CONSUMER_SUBSCRIBE_MATCHER.isTrueFor(call.callee(), ctx);
  }

  private static boolean isDunderMethod(FunctionDef functionDef) {
    String methodName = functionDef.name().name();
    return methodName.startsWith("__");
  }

  private static boolean isTrivialFunction(StatementList body, SubscriptionContext ctx) {
    for (Statement statement : body.statements()) {
      if (!CheckUtils.isEmptyStatement(statement) && !statement.is(Tree.Kind.RAISE_STMT) && !isReturnNotImplemented(statement, ctx)) {
        return false;
      }
    }
    return true;
  }

  private static boolean isReturnNotImplemented(Statement statement, SubscriptionContext ctx) {
    return statement.is(Tree.Kind.RETURN_STMT) &&
      ((ReturnStatement) statement).expressions().stream().allMatch(e -> NOT_IMPLEMENTED_MATCHER.isTrueFor(e, ctx));
  }

  private static boolean mightBeOverridingMethod(FunctionDef functionDef) {
    FunctionType functionType = (FunctionType) functionDef.name().typeV2();
    return functionType.owner() instanceof ClassType classType && (classType.hasUnresolvedHierarchy() || classType.inheritedMember(functionType.name()).isPresent());
  }

  private static class AsyncFeatureVisitor extends BaseTreeVisitor {

    private boolean asyncFeatureFound = false;

    public boolean hasAsyncFeature() {
      return asyncFeatureFound;
    }

    @Override
    public void visitAwaitExpression(AwaitExpression awaitExpression) {
      asyncFeatureFound = true;
    }

    @Override
    public void visitForStatement(ForStatement forStatement) {
      if (forStatement.isAsync()) {
        asyncFeatureFound = true;
        return;
      }
      if (!asyncFeatureFound) {
        super.visitForStatement(forStatement);
      }
    }

    @Override
    public void visitWithStatement(WithStatement withStatement) {
      if (withStatement.isAsync()) {
        asyncFeatureFound = true;
      }
      if (!asyncFeatureFound) {
        super.visitWithStatement(withStatement);
      }
    }

    @Override
    public void visitYieldStatement(YieldStatement yieldStatement) {
      asyncFeatureFound = true;
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpression) {
      asyncFeatureFound = true;
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Skip nested functions
    }

    @Override
    public void visitComprehensionFor(ComprehensionFor tree) {
      asyncFeatureFound |= tree.asyncToken() != null;
      super.visitComprehensionFor(tree);
    }

    @Override
    protected void scan(@Nullable Tree tree) {
      // Stop scanning if we've already found an async feature
      if (!asyncFeatureFound && tree != null) {
        tree.accept(this);
      }
    }
  }
}
