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
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8410")
public class FastAPIDependencyAnnotatedCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use \"Annotated\" type hints for FastAPI dependency injection";

  private static final Set<String> ROUTES = Set.of(
    "get", "post", "put", "delete", "patch", "options", "head", "trace"
  );

  private static final String FASTAPI_MODULE = "fastapi.applications.FastAPI";
  private static final String API_ROUTER_MODULE = "fastapi.routing.APIRouter";

  private static final TypeMatcher FASTAPI_ROUTE_METHODS_MATCHER = TypeMatchers.any(
    Stream.concat(
      ROUTES.stream().map(r -> TypeMatchers.isType(FASTAPI_MODULE + "." + r)),
      ROUTES.stream().map(r -> TypeMatchers.isType(API_ROUTER_MODULE + "." + r))
    )
  );

  private static final TypeMatcher FASTAPI_DEPENDENCY_FUNCTIONS_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("fastapi.param_functions.Depends"),
    TypeMatchers.isType("fastapi.param_functions.Query"),
    TypeMatchers.isType("fastapi.param_functions.Path"),
    TypeMatchers.isType("fastapi.param_functions.Body"),
    TypeMatchers.isType("fastapi.param_functions.Header"),
    TypeMatchers.isType("fastapi.param_functions.Cookie"),
    TypeMatchers.isType("fastapi.param_functions.Form"),
    TypeMatchers.isType("fastapi.param_functions.File")
  );

  private static final TypeMatcher TYPING_ANNOTATED_MATCHER = TypeMatchers.isType("typing.Annotated");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FastAPIDependencyAnnotatedCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    if (!hasFastAPIRouteDecorator(functionDef, ctx)) {
      return;
    }

    ParameterList parameterList = functionDef.parameters();
    if (parameterList == null) {
      return;
    }

    parameterList.nonTuple().stream()
      .filter(param -> isParameterUsingOldDependencySyntax(param, ctx))
      .forEach(param -> ctx.addIssue(param, MESSAGE));
  }

  private static boolean hasFastAPIRouteDecorator(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .map(Decorator::expression)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(CallExpression.class))
      .anyMatch(callExpr -> FASTAPI_ROUTE_METHODS_MATCHER.isTrueFor(callExpr.callee(), ctx));
  }

  private static boolean isParameterUsingOldDependencySyntax(Parameter param, SubscriptionContext ctx) {
    Expression defaultValue = param.defaultValue();
    if (!(defaultValue instanceof CallExpression callExpr)) {
      return false;
    }

    if (!FASTAPI_DEPENDENCY_FUNCTIONS_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      return false;
    }

    TypeAnnotation typeAnnotation = param.typeAnnotation();
    return !isUsingAnnotatedWithDependency(typeAnnotation, ctx);
  }

  private static boolean isUsingAnnotatedWithDependency(@Nullable TypeAnnotation typeAnnotation, SubscriptionContext ctx) {
    if (typeAnnotation == null) {
      return false;
    }
    Expression annotationExpr = typeAnnotation.expression();
    if (annotationExpr instanceof SubscriptionExpression subscriptionExpr) {
      Expression object = subscriptionExpr.object();
      if (object instanceof Name name && TYPING_ANNOTATED_MATCHER.isTrueFor(name, ctx)) {
        return subscriptionExpr.subscripts().expressions().stream()
          .anyMatch(expr -> {
            if (expr instanceof CallExpression callExpr) {
              return FASTAPI_DEPENDENCY_FUNCTIONS_MATCHER.isTrueFor(callExpr.callee(), ctx);
            }
            return false;
          });
      }
    }
    return false;
  }
}
