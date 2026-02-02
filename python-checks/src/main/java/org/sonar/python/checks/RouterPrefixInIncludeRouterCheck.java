/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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

import java.util.Optional;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8413")
public class RouterPrefixInIncludeRouterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE =
    "Define the prefix in the \"APIRouter\" constructor instead of in \"include_router()\".";

  private static final TypeMatcher INCLUDE_ROUTER_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("fastapi.applications.FastAPI.include_router"),
    TypeMatchers.isType("fastapi.routing.APIRouter.include_router")
  );

  private static final TypeMatcher API_ROUTER_MATCHER = TypeMatchers.isType("fastapi.routing.APIRouter");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, RouterPrefixInIncludeRouterCheck::checkIncludeRouterCall);
  }

  private static void checkIncludeRouterCall(SubscriptionContext ctx) {
    CallExpression callExpr = (CallExpression) ctx.syntaxNode();

    if (!INCLUDE_ROUTER_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      return;
    }

    RegularArgument prefixArg = TreeUtils.argumentByKeyword("prefix", callExpr.arguments());
    if (prefixArg == null) {
      return;
    }

    Optional<RegularArgument> routerArg = TreeUtils.nthArgumentOrKeywordOptional(0, "router", callExpr.arguments());
    if (routerArg.isEmpty()) {
      return;
    }

    Expression routerExpr = routerArg.get().expression();
    if (!(routerExpr instanceof Name routerName)) {
      return;
    }

    SymbolV2 routerSymbol = routerName.symbolV2();
    if (routerSymbol == null) {
      return;
    }

    if (isRouterWithoutPrefixInConstructor(routerSymbol, ctx)) {
      var keyword = prefixArg.keywordArgument();
      if (keyword != null) {
        ctx.addIssue(keyword, MESSAGE);
      }
    }
  }

  private static boolean isRouterWithoutPrefixInConstructor(SymbolV2 routerSymbol, SubscriptionContext ctx) {
    return routerSymbol.usages().stream()
      .filter(UsageV2::isBindingUsage)
      .map(UsageV2::tree)
      .flatMap(RouterPrefixInIncludeRouterCheck::getBindingCallExpression)
      .filter(call -> API_ROUTER_MATCHER.isTrueFor(call.callee(), ctx))
      .anyMatch(RouterPrefixInIncludeRouterCheck::callHasNoPrefixArgument);
  }

  private static Stream<CallExpression> getBindingCallExpression(Tree bindingTree) {
    if (!(bindingTree instanceof Name name)) {
      return Stream.empty();
    }
    return Expressions.singleAssignedNonNameValue(name).stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(CallExpression.class));
  }

  private static boolean callHasNoPrefixArgument(CallExpression call) {
    return TreeUtils.argumentByKeyword("prefix", call.arguments()) == null;
  }
}
