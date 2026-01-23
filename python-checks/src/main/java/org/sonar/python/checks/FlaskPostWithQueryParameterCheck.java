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
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8370")
public class FlaskPostWithQueryParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Do not use query parameters with POST requests; use path parameters or request body instead.";
  private static final String FLASK_ROUTE_FQN = "flask.app.Flask.route";
  private static final String BLUEPRINT_ROUTE_FQN = "flask.blueprints.Blueprint.route";
  private static final String REQUEST_FQN = "flask.wrappers.Request";

  private static final List<String> SAFE_VERBS = List.of("GET", "HEAD", "OPTIONS", "TRACE");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FlaskPostWithQueryParameterCheck::checkPostWithQueryParams);
  }

  private static void checkPostWithQueryParams(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (!isPostRoute(functionDef, ctx)) {
      return;
    }

    TreeUtils.firstChild(functionDef, tree -> tree instanceof QualifiedExpression qe && isRequestArgs(qe, ctx))
      .ifPresent(requestArgs -> ctx.addIssue(requestArgs, MESSAGE));
  }

  private static boolean isPostRoute(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .anyMatch(decorator -> isPostDecorator(decorator, ctx));
  }

  private static boolean isPostDecorator(Decorator decorator, SubscriptionContext ctx) {
    Expression expression = decorator.expression();
    if (!(expression instanceof CallExpression callExpr)) {
      return false;
    }
    if (!isFlaskRouteDecorator(callExpr, ctx)) {
      return false;
    }

    return getMethodsArgument(callExpr)
      .map(FlaskPostWithQueryParameterCheck::containsPostButNoSafeVerbs)
      .orElse(false);
  }

  private static boolean isFlaskRouteDecorator(CallExpression callExpr, SubscriptionContext ctx) {
    return TypeMatchers.any(TypeMatchers.isType(FLASK_ROUTE_FQN), TypeMatchers.isType(BLUEPRINT_ROUTE_FQN))
      .isTrueFor(callExpr.callee(), ctx);
  }

  private static Optional<List<Expression>> getMethodsArgument(CallExpression callExpr) {
    RegularArgument methodsArg = TreeUtils.argumentByKeyword("methods", callExpr.arguments());
    if (methodsArg == null) {
      return Optional.empty();
    }

    Expression argExpr = methodsArg.expression();
    if (!argExpr.is(Tree.Kind.LIST_LITERAL)) {
      return Optional.empty();
    }

    ListLiteral listLiteral = (ListLiteral) argExpr;
    return Optional.of(listLiteral.elements().expressions());
  }

  private static boolean containsPostButNoSafeVerbs(List<Expression> methods) {
    return getStringValuesFromExpressions(methods)
      .anyMatch("POST"::equalsIgnoreCase) &&
      getStringValuesFromExpressions(methods)
        .allMatch(verb -> !SAFE_VERBS.contains(verb.toUpperCase(Locale.ROOT)));
  }

  private static Stream<String> getStringValuesFromExpressions(List<Expression> expressions) {
    return expressions.stream()
      .filter(expr -> expr.is(Tree.Kind.STRING_LITERAL))
      .map(StringLiteral.class::cast)
      .map(StringLiteral::trimmedQuotesValue);

  }

  private static boolean isRequestArgs(QualifiedExpression qualifiedExpr, SubscriptionContext ctx) {
    if (!"args".equals(qualifiedExpr.name().name())) {
      return false;
    }

    if (!(qualifiedExpr.qualifier() instanceof Name request)) {
      return false;
    }

    return TypeMatchers.isObjectOfType(REQUEST_FQN).isTrueFor(request, ctx);
  }
}
