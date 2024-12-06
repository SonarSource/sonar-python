/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.checks.hotspots;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.tree.FunctionDefImpl;

import static org.sonar.plugins.python.api.tree.Tree.Kind.CALL_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.FUNCDEF;
import static org.sonar.plugins.python.api.tree.Tree.Kind.LIST_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.REGULAR_ARGUMENT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.STRING_LITERAL;
import static org.sonar.python.tree.TreeUtils.argumentByKeyword;
import static org.sonar.python.tree.TreeUtils.getSymbolFromTree;

@Rule(key = "S3752")
public class UnsafeHttpMethodsCheck extends PythonSubscriptionCheck {

  private static final Set<String> SAFE_HTTP_METHODS = new HashSet<>(Arrays.asList("GET", "HEAD", "OPTIONS"));
  private static final Set<String> UNSAFE_HTTP_METHODS = new HashSet<>(Arrays.asList("POST", "PUT", "DELETE"));
  private static final Set<String> COMPLIANT_DECORATORS = new HashSet<>(Arrays.asList(
    "django.views.decorators.http.require_POST",
    "django.views.decorators.http.require_GET",
    "django.views.decorators.http.require_safe"
  ));
  private static final String MESSAGE = "Make sure allowing safe and unsafe HTTP methods is safe here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (isDjangoView(functionDef)) {
        checkDjangoView(functionDef, ctx);
      } else {
        getFlaskViewDecorator(functionDef).ifPresent(callExpression -> checkFlaskView(callExpression, ctx));
      }
    });
  }

  private static void checkDjangoView(FunctionDef functionDef, SubscriptionContext ctx) {
    for (Decorator decorator : functionDef.decorators()) {
      if (getSymbolFromTree(decorator.expression())
        .filter(symbol -> symbol.fullyQualifiedName() == null || COMPLIANT_DECORATORS.contains(symbol.fullyQualifiedName()))
        .isPresent()) {
        return;
      }
      if (decorator.expression().is(CALL_EXPR)) {
        CallExpression callExpression = (CallExpression) decorator.expression();
        Symbol symbol = callExpression.calleeSymbol();
        if (symbol != null && "django.views.decorators.http.require_http_methods".equals(symbol.fullyQualifiedName())) {
          checkRequireHttpMethodsDecorator(ctx, callExpression);
          return;
        }
      }
    }
    ctx.addIssue(functionDef.name(), MESSAGE);
  }

  private static void checkRequireHttpMethodsDecorator(SubscriptionContext ctx, CallExpression callExpression) {
    List<Argument> arguments = callExpression.arguments();
    if (!arguments.isEmpty() && hasBothUnsafeAndSafeHttpMethods(arguments.get(0))) {
      ctx.addIssue(callExpression, MESSAGE);
    }
  }

  private static boolean hasBothUnsafeAndSafeHttpMethods(Argument argument) {
    boolean hasSafeHttpMethod = false;
    boolean hasUnsafeHttpMethod = false;
    if (argument.is(REGULAR_ARGUMENT) && ((RegularArgument) argument).expression().is(LIST_LITERAL)) {
      ListLiteral listLiteral = (ListLiteral) ((RegularArgument) argument).expression();
      for (Expression expression : listLiteral.elements().expressions()) {
        if (expression.is(STRING_LITERAL)) {
          String value = ((StringLiteral) expression).trimmedQuotesValue();
          if (SAFE_HTTP_METHODS.contains(value)) {
            hasSafeHttpMethod = true;
          } else if (UNSAFE_HTTP_METHODS.contains(value)) {
            hasUnsafeHttpMethod = true;
          }
        }
      }
    }
    return hasSafeHttpMethod && hasUnsafeHttpMethod;
  }

  private static boolean isDjangoView(FunctionDef functionDef) {
    FunctionSymbol functionSymbol = ((FunctionDefImpl) functionDef).functionSymbol();
    return Optional.ofNullable(functionSymbol)
      .map(FunctionSymbolImpl.class::cast)
      .filter(FunctionSymbolImpl::isDjangoView)
      .isPresent();
  }

  private static Optional<CallExpression> getFlaskViewDecorator(FunctionDef functionDef) {
    return functionDef.decorators().stream()
      .map(Decorator::expression)
      .filter(expression -> expression.is(CALL_EXPR))
      .map(CallExpression.class::cast)
      .filter(UnsafeHttpMethodsCheck::isFlaskRouteDecorator)
      .findFirst();
  }

  private static boolean isFlaskRouteDecorator(CallExpression callExpression) {
    Symbol calleeSymbol = callExpression.calleeSymbol();
    return calleeSymbol != null && "flask.scaffold.Scaffold.route".equals(calleeSymbol.fullyQualifiedName());
  }

  private static void checkFlaskView(CallExpression callExpression, SubscriptionContext ctx) {
    RegularArgument methodsArg = argumentByKeyword("methods", callExpression.arguments());
    if (methodsArg != null && hasBothUnsafeAndSafeHttpMethods(methodsArg)) {
      ctx.addIssue(callExpression, MESSAGE);
    }
  }
}
