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

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8411")
public class FastAPIPathParametersCheck extends PythonSubscriptionCheck {

  private static final String MISSING_PARAM_MESSAGE = "Add path parameter \"%s\" to the function signature.";
  private static final String POSITIONAL_ONLY_MESSAGE = "Path parameter \"%s\" should not be positional-only.";

  private static final List<String> HTTP_METHODS = List.of(
    "get", "post", "put", "delete", "patch", "options", "head", "trace");

  private static final Pattern PATH_PARAM_PATTERN = Pattern.compile("\\{([^}:]+)(?::[^}]*)?\\}");

  private static final TypeMatcher FASTAPI_ROUTE_MATCHER = TypeMatchers.any(
    HTTP_METHODS.stream()
      .flatMap(method -> Stream.of(
        TypeMatchers.isType("fastapi.FastAPI." + method),
        TypeMatchers.isType("fastapi.APIRouter." + method)))
  );

  private record FunctionParameterInfo(Set<String> allParams, Set<String> positionalOnlyParams, boolean hasVariadicKeyword) {
    static FunctionParameterInfo empty() {
      return new FunctionParameterInfo(Set.of(), Set.of(), false);
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FastAPIPathParametersCheck::checkFunction);
  }

  private static void checkFunction(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    for (Decorator decorator : functionDef.decorators()) {
      checkDecorator(ctx, decorator, functionDef);
    }
  }

  private static void checkDecorator(SubscriptionContext ctx, Decorator decorator, FunctionDef functionDef) {
    Expression expr = decorator.expression();
    if (!(expr instanceof CallExpression callExpr)) {
      return;
    }

    if (!FASTAPI_ROUTE_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      return;
    }

    Set<String> pathParams = extractPathParameters(callExpr);
    if (pathParams.isEmpty()) {
      return;
    }

    FunctionParameterInfo paramInfo = extractFunctionParameters(functionDef);
    reportIssues(ctx, functionDef, pathParams, paramInfo);
  }

  private static Set<String> extractPathParameters(CallExpression callExpr) {
    Set<String> pathParams = new HashSet<>();
    String pathString = getPathArgument(callExpr).orElse("");
    Matcher matcher = PATH_PARAM_PATTERN.matcher(pathString);
    while (matcher.find()) {
      pathParams.add(matcher.group(1));
    }
    return pathParams;
  }

  private static Optional<String> getPathArgument(CallExpression callExpr) {
    return TreeUtils.nthArgumentOrKeywordOptional(0, "path", callExpr.arguments())
      .flatMap(arg -> extractStringValue(arg.expression()));
  }

  private static Optional<String> extractStringValue(Expression expression) {
    return Optional.ofNullable(Expressions.extractStringLiteral(expression))
      .map(Expressions::unescape);
  }

  private static FunctionParameterInfo extractFunctionParameters(FunctionDef functionDef) {
    return getFunctionType(functionDef)
      .map(FastAPIPathParametersCheck::buildParameterInfo)
      .orElse(FunctionParameterInfo.empty());
  }

  private static Optional<FunctionType> getFunctionType(FunctionDef functionDef) {
    PythonType functionType = functionDef.name().typeV2();
    if (functionType instanceof FunctionType funcType) {
      return Optional.of(funcType);
    }
    return Optional.empty();
  }

  private static FunctionParameterInfo buildParameterInfo(FunctionType functionType) {
    Set<String> allParams = new HashSet<>();
    Set<String> positionalOnlyParams = new HashSet<>();
    boolean hasVariadicKeyword = functionType.parameters().stream()
      .anyMatch(param -> param.isVariadic() && param.isKeywordVariadic());

    functionType.parameters().stream()
      .filter(param -> !param.isVariadic())
      .forEach(param -> addParameter(param, allParams, positionalOnlyParams));

    return new FunctionParameterInfo(allParams, positionalOnlyParams, hasVariadicKeyword);
  }

  private static void addParameter(ParameterV2 param, Set<String> allParams, Set<String> positionalOnlyParams) {
    String paramName = param.name();
    if (paramName != null) {
      allParams.add(paramName);
      if (param.isPositionalOnly()) {
        positionalOnlyParams.add(paramName);
      }
    }
  }

  private static void reportIssues(SubscriptionContext ctx, FunctionDef functionDef, Set<String> pathParams, FunctionParameterInfo paramInfo) {
    pathParams.stream()
      .filter(param -> isMissingFromSignature(param, paramInfo))
      .forEach(param -> ctx.addIssue(functionDef.name(), String.format(MISSING_PARAM_MESSAGE, param)));

    pathParams.stream()
      .filter(paramInfo.positionalOnlyParams::contains)
      .forEach(param -> ctx.addIssue(functionDef.name(), String.format(POSITIONAL_ONLY_MESSAGE, param)));
  }

  private static boolean isMissingFromSignature(String pathParam, FunctionParameterInfo paramInfo) {
    return !paramInfo.allParams.contains(pathParam) && !paramInfo.hasVariadicKeyword;
  }
}
