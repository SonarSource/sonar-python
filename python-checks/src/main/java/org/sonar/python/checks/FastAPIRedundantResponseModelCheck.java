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

import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8409")
public class FastAPIRedundantResponseModelCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE =
    "Remove this redundant \"response_model\" parameter; it duplicates the return type annotation.";

  private static final String FASTAPI_MODULE = "fastapi.applications.FastAPI";
  private static final String API_ROUTER_MODULE = "fastapi.routing.APIRouter";
  private static final Set<String> ROUTES = Set.of(
    "get", "post", "put", "delete", "patch", "options", "head", "trace"
  );

  private static final TypeMatcher FASTAPI_METHODS_MATCHER = TypeMatchers.any(
    Stream.concat(
      ROUTES.stream().map(methodName -> TypeMatchers.isType(FASTAPI_MODULE + "." + methodName)),
      ROUTES.stream().map(methodName -> TypeMatchers.isType(API_ROUTER_MODULE + "." + methodName)))
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FastAPIRedundantResponseModelCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    TypeAnnotation returnTypeAnnotation = functionDef.returnTypeAnnotation();

    if (returnTypeAnnotation == null) {
      return;
    }

    for (Decorator decorator : functionDef.decorators()) {
      checkDecorator(ctx, decorator, returnTypeAnnotation);
    }
  }

  private static void checkDecorator(SubscriptionContext ctx, Decorator decorator, TypeAnnotation returnTypeAnnotation) {
    CallExpression callExpr = getDecoratorCallExpression(decorator);
    if (callExpr == null || !isFastApiRouteDecorator(callExpr, ctx)) {
      return;
    }

    RegularArgument responseModelArg = TreeUtils.argumentByKeyword("response_model", callExpr.arguments());
    if (responseModelArg == null) {
      return;
    }

    Expression responseModelExpr = responseModelArg.expression();
    Expression returnTypeExpr = returnTypeAnnotation.expression();

    if (CheckUtils.areEquivalent(responseModelExpr, returnTypeExpr)) {
      reportRedundantResponseModel(ctx, responseModelArg);
    }
  }

  @CheckForNull
  private static CallExpression getDecoratorCallExpression(Decorator decorator) {
    Expression decoratorExpr = decorator.expression();
    return decoratorExpr instanceof CallExpression callExpr ? callExpr : null;
  }

  private static boolean isFastApiRouteDecorator(CallExpression callExpr, SubscriptionContext ctx) {
    return FASTAPI_METHODS_MATCHER.isTrueFor(callExpr.callee(), ctx);
  }

  private static void reportRedundantResponseModel(SubscriptionContext ctx, RegularArgument responseModelArg) {
    var keyword = responseModelArg.keywordArgument();
    if (keyword != null) {
      ctx.addIssue(keyword.firstToken(), responseModelArg.expression().lastToken(), MESSAGE);
    }
  }
}
