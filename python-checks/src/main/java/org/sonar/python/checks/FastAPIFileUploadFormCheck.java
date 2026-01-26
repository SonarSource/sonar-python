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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8389")
public class FastAPIFileUploadFormCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_BODY_WITH_FILE =
    "Use \"Form()\" instead of \"Body()\" when handling file uploads; " +
    "\"Body()\" expects JSON, which is incompatible with multipart/form-data.";

  private static final String MESSAGE_DEPENDS_WITH_FILE =
    "Use \"Form()\" with Pydantic validators instead of \"Depends()\" for " +
    "file upload endpoints; query parameters may expose sensitive data in URLs.";

  private static final TypeMatcher IS_FASTAPI_ROUTE = TypeMatchers.any(
    TypeMatchers.isType("fastapi.routing.APIRouter.post"),
    TypeMatchers.isType("fastapi.routing.APIRouter.put"),
    TypeMatchers.isType("fastapi.routing.APIRouter.patch"),
    TypeMatchers.isType("fastapi.routing.APIRouter.delete"),
    TypeMatchers.isType("fastapi.applications.FastAPI.post"),
    TypeMatchers.isType("fastapi.applications.FastAPI.put"),
    TypeMatchers.isType("fastapi.applications.FastAPI.patch"),
    TypeMatchers.isType("fastapi.applications.FastAPI.delete")
  );

  private static final TypeMatcher IS_FILE_PARAM = TypeMatchers.isType("fastapi.param_functions.File");
  private static final TypeMatcher IS_BODY_PARAM = TypeMatchers.isType("fastapi.param_functions.Body");
  private static final TypeMatcher IS_DEPENDS_PARAM = TypeMatchers.isType("fastapi.param_functions.Depends");

  private static final TypeMatcher IS_UPLOAD_FILE = TypeMatchers.any(
    TypeMatchers.isOrExtendsType("fastapi.datastructures.UploadFile"),
    TypeMatchers.isOrExtendsType("starlette.datastructures.UploadFile")
  );

  private static final TypeMatcher IS_PYDANTIC_MODEL = TypeMatchers.isOrExtendsType("pydantic.BaseModel");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FastAPIFileUploadFormCheck::checkFunctionDef);
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
    List<Parameter> parameters = parameterList.nonTuple();

    boolean hasFileParameter = parameters.stream()
      .anyMatch(param -> hasFileParam(param, ctx));

    if (!hasFileParameter) {
      return;
    }

    parameters.stream()
      .filter(param -> hasBodyParam(param, ctx))
      .forEach(param -> ctx.addIssue(param, MESSAGE_BODY_WITH_FILE));

    parameters.stream()
      .filter(param -> hasDependsWithPydanticModel(param, ctx))
      .forEach(param -> ctx.addIssue(param, MESSAGE_DEPENDS_WITH_FILE));
  }

  private static boolean hasFastAPIRouteDecorator(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .map(Decorator::expression)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(CallExpression.class))
      .anyMatch(callExpr -> IS_FASTAPI_ROUTE.isTrueFor(callExpr.callee(), ctx));
  }

  private static boolean hasFileParam(Parameter param, SubscriptionContext ctx) {
    Expression defaultValue = param.defaultValue();
    if (defaultValue instanceof CallExpression callExpr && IS_FILE_PARAM.isTrueFor(callExpr.callee(), ctx)) {
      return true;
    }

    TypeAnnotation annotation = param.typeAnnotation();
    if (annotation != null) {
      Expression annotationExpr = annotation.expression();

      if (IS_UPLOAD_FILE.isTrueFor(annotationExpr, ctx)) {
        return true;
      }

      if (annotationExpr instanceof SubscriptionExpression subscriptionExpr) {
        return subscriptionExpr.subscripts().expressions().stream()
          .anyMatch(expr -> IS_UPLOAD_FILE.isTrueFor(expr, ctx));
      }
    }

    return false;
  }

  private static boolean hasBodyParam(Parameter param, SubscriptionContext ctx) {
    Expression defaultValue = param.defaultValue();
    if (defaultValue instanceof CallExpression callExpr) {
      return IS_BODY_PARAM.isTrueFor(callExpr.callee(), ctx);
    }
    return false;
  }

  private static boolean hasDependsWithPydanticModel(Parameter param, SubscriptionContext ctx) {
    Expression defaultValue = param.defaultValue();
    if (!(defaultValue instanceof CallExpression dependsCall)) {
      return false;
    }

    if (!IS_DEPENDS_PARAM.isTrueFor(dependsCall.callee(), ctx)) {
      return false;
    }

    if (!dependsCall.arguments().isEmpty()) {
      var firstArg = dependsCall.arguments().get(0);
      if (firstArg instanceof RegularArgument regularArg) {
        Expression argExpr = regularArg.expression();
        return IS_PYDANTIC_MODEL.isTrueFor(argExpr, ctx);
      }
    }

    TypeAnnotation annotation = param.typeAnnotation();
    if (annotation != null) {
      Expression annotationExpr = annotation.expression();
      return IS_PYDANTIC_MODEL.isTrueFor(annotationExpr, ctx);
    }

    return false;
  }
}
