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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8414")
public class CorsMiddlewareOrderingCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add CORSMiddleware last in the middleware chain.";

  // Stubs are missing entirely for fastapi/starlette.middleware.cors so we match by FQN for add_middleware
  private static final TypeMatcher ADD_MIDDLEWARE_MATCHER = TypeMatchers.any(
    TypeMatchers.withFQN("fastapi.applications.FastAPI.add_middleware"),
    TypeMatchers.withFQN("starlette.applications.Starlette.add_middleware")
  );

  private static final TypeMatcher CORS_MIDDLEWARE_MATCHER = TypeMatchers.any(
    TypeMatchers.withFQN("fastapi.middleware.cors.CORSMiddleware"),
    TypeMatchers.withFQN("starlette.middleware.cors.CORSMiddleware")
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, CorsMiddlewareOrderingCheck::checkAddMiddlewareCall);
  }

  private static void checkAddMiddlewareCall(SubscriptionContext ctx) {
    CallExpression callExpr = (CallExpression) ctx.syntaxNode();

    if (!ADD_MIDDLEWARE_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      return;
    }

    if (!isCorsMiddleware(ctx, callExpr)) {
      return;
    }

    SymbolV2 receiverSymbol = extractReceiverSymbol(callExpr).orElse(null);
    if (receiverSymbol == null) {
      return;
    }

    Tree currentEnclosingScope = getScope(callExpr);
    if (currentEnclosingScope == null) {
      return;
    }

    int currentLine = callExpr.firstToken().line();

    if (hasSubsequentMiddlewareAddition(ctx, receiverSymbol, currentLine, currentEnclosingScope)) {
      ctx.addIssue(callExpr.callee(), MESSAGE);
    }
  }

  private static boolean isCorsMiddleware(SubscriptionContext ctx, CallExpression callExpr) {
    return TreeUtils.nthArgumentOrKeywordOptional(0, "middleware_class", callExpr.arguments())
      .map(RegularArgument::expression)
      .map(expr -> {
        if (CORS_MIDDLEWARE_MATCHER.isTrueFor(expr, ctx)) {
          return true;
        }
        if (expr instanceof Name name) {
          Expression assignedValue = Expressions.singleAssignedValue(name);
          return assignedValue != null && CORS_MIDDLEWARE_MATCHER.isTrueFor(assignedValue, ctx);
        }
        return false;
      })
      .orElse(false);
  }

  private static boolean hasSubsequentMiddlewareAddition(
    SubscriptionContext ctx,
    SymbolV2 receiverSymbol,
    int currentLine,
    Tree currentEnclosingScope) {

    return receiverSymbol.usages().stream()
      .filter(usage -> usage.kind() == UsageV2.Kind.OTHER)
      .filter(usage -> usage.tree().firstToken().line() > currentLine)
      .filter(usage -> isSameScope(usage.tree(), currentEnclosingScope))
      .anyMatch(usage -> isReceiverOfAddMiddleware(usage, ctx));
  }

  private static boolean isSameScope(Tree usageTree, Tree currentEnclosingScope) {
    Tree usageEnclosingScope = getScope(usageTree);
    return usageEnclosingScope == currentEnclosingScope;
  }

  private static Tree getScope(Tree tree) {
    return TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA, Tree.Kind.FILE_INPUT);
  }

  private static boolean isReceiverOfAddMiddleware(UsageV2 usage, SubscriptionContext ctx) {
    Tree tree = usage.tree();

    Tree parent = tree.parent();
    if (!(parent instanceof QualifiedExpression qualifiedExpr)) {
      return false;
    }

    Tree qualiifiedParent = qualifiedExpr.parent();
    if (!(qualiifiedParent instanceof CallExpression parentCall)) {
      return false;
    }

    return ADD_MIDDLEWARE_MATCHER.isTrueFor(parentCall.callee(), ctx);
  }

  private static Optional<SymbolV2> extractReceiverSymbol(CallExpression callExpr) {
    return Optional.ofNullable(callExpr.callee())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::qualifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::symbolV2);
  }
}
