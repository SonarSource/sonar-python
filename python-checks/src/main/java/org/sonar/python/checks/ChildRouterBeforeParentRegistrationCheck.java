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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8401")
public class ChildRouterBeforeParentRegistrationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Include child routers before registering the parent router.";

  private static final TypeMatcher INCLUDE_ROUTER_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("fastapi.applications.FastAPI.include_router"),
    TypeMatchers.isType("fastapi.routing.APIRouter.include_router")
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ChildRouterBeforeParentRegistrationCheck::checkIncludeRouterCall);
  }

  private static void checkIncludeRouterCall(SubscriptionContext ctx) {
    CallExpression callExpr = (CallExpression) ctx.syntaxNode();

    if (!INCLUDE_ROUTER_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      return;
    }

    SymbolV2 receiverSymbol = extractReceiverSymbol(callExpr).orElse(null);
    if (receiverSymbol == null) {
      return;
    }

    Tree currentEnclosingScope = getScope(callExpr);
    // defensive check
    if (currentEnclosingScope == null) {
      return;
    }

    int currentLine = callExpr.firstToken().line();

    if (wasReceiverPreviouslyRegistered(ctx, receiverSymbol, currentLine, currentEnclosingScope)) {
      ctx.addIssue(callExpr, MESSAGE);
    }
  }

  private static boolean wasReceiverPreviouslyRegistered(
    SubscriptionContext ctx,
    SymbolV2 receiverSymbol,
    int currentLine,
    Tree currentEnclosingScope) {

    return receiverSymbol.usages().stream()
      .filter(usage -> usage.kind() == UsageV2.Kind.OTHER)
      .filter(usage -> usage.tree().firstToken().line() < currentLine)
      .filter(usage -> isSameScope(usage.tree(), currentEnclosingScope))
      .anyMatch(usage -> isArgumentToIncludeRouter(usage, ctx));
  }


  private static boolean isSameScope(Tree usageTree, Tree currentEnclosingFunction) {
    Tree usageEnclosingFunction = getScope(usageTree);
    return usageEnclosingFunction == currentEnclosingFunction;
  }

  private static Tree getScope(Tree tree) {
    return TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA, Tree.Kind.FILE_INPUT);
  }

  private static boolean isArgumentToIncludeRouter(UsageV2 usage, SubscriptionContext ctx) {
    Tree tree = usage.tree();
    RegularArgument arg = TreeUtils.firstAncestorOfClass(tree, RegularArgument.class);
    if (arg == null) {
      return false;
    }

    CallExpression parentCall = TreeUtils.firstAncestorOfClass(arg, CallExpression.class);
    if (parentCall == null) {
      return false;
    }

    return INCLUDE_ROUTER_MATCHER.isTrueFor(parentCall.callee(), ctx)
      && isFirstRouterArgument(parentCall, arg);
  }

  private static boolean isFirstRouterArgument(CallExpression call, RegularArgument arg) {
    return TreeUtils.nthArgumentOrKeywordOptional(0, "router", call.arguments())
      .map(firstArg -> firstArg == arg)
      .orElse(false);
  }

  private static Optional<SymbolV2> extractReceiverSymbol(CallExpression callExpr) {
    return Optional.ofNullable(callExpr.callee())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::qualifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::symbolV2);
  }
}
