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
package org.sonar.python.checks.hotspots;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.tree.FunctionDefImpl;

@Rule(key = "S3752")
public class UnsafeHttpMethodsCheck extends PythonSubscriptionCheck {

  private static final TypeMatcher COMPLIANT_DECORATORS = TypeMatchers.any(
    TypeMatchers.isType("django.views.decorators.http.require_POST"),
    TypeMatchers.isType("django.views.decorators.http.require_GET"),
    TypeMatchers.isType("django.views.decorators.http.require_safe"));

  private static final TypeMatcher IS_REQUIRED_HTTP_METHOD = TypeMatchers.isType("django.views.decorators.http.require_http_methods");

  private static final String MESSAGE = "Explicitly specify the HTTP methods this endpoint accepts.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (isDjangoView(functionDef)) {
        checkDjangoView(functionDef, ctx);
      }
    });
  }

  private static void checkDjangoView(FunctionDef functionDef, SubscriptionContext ctx) {
    boolean hasCompliantDecorator = functionDef.decorators().stream()
      .map(Decorator::expression)
      .anyMatch(expr -> COMPLIANT_DECORATORS.isTrueFor(expr, ctx) || isCallRequiredHTTPMethod(expr, ctx));

    boolean hasNonDjangoDecorator = functionDef.decorators().stream()
      .map(Decorator::expression)
      .anyMatch(expr -> !isKnownDjangoDecorator(expr));

    if (!hasCompliantDecorator && !hasNonDjangoDecorator && !hasRequestMethodCheck(functionDef)) {
      ctx.addIssue(functionDef.name(), MESSAGE);
    }
  }

  private static boolean isCallRequiredHTTPMethod(Expression expression, SubscriptionContext ctx) {
    return expression instanceof CallExpression callExpression &&
      IS_REQUIRED_HTTP_METHOD.isTrueFor(callExpression.callee(), ctx);
  }

  private static boolean isKnownDjangoDecorator(Expression expression) {
    var type = expression instanceof CallExpression callExpr
      ? callExpr.callee().typeV2()
      : expression.typeV2();
    return type instanceof UnknownType.UnresolvedImportType uit &&
      uit.importPath().startsWith("django.");
  }

  private static boolean hasRequestMethodCheck(FunctionDef functionDef) {
    var params = functionDef.parameters();
    if (params == null || params.nonTuple().isEmpty()) {
      return false;
    }
    Name firstParam = params.nonTuple().get(0).name();
    if (firstParam == null) {
      return false;
    }
    SymbolV2 symbol = firstParam.symbolV2();
    if (symbol == null) {
      return false;
    }
    return symbol.usages().stream()
      .filter(usage -> usage.kind() != UsageV2.Kind.PARAMETER)
      .anyMatch(UnsafeHttpMethodsCheck::isRequestMethodInIfCondition);
  }

  private static boolean isRequestMethodInIfCondition(UsageV2 usage) {
    if (!(usage.tree().parent() instanceof QualifiedExpression qe) || !"method".equals(qe.name().name())) {
      return false;
    }
    if (!(qe.parent() instanceof BinaryExpression comparison) ||
      (!comparison.is(Tree.Kind.COMPARISON) && !comparison.is(Tree.Kind.IN))) {
      return false;
    }
    Tree ancestor = comparison.parent();
    while (ancestor instanceof BinaryExpression || ancestor instanceof ParenthesizedExpression) {
      ancestor = ancestor.parent();
    }
    return ancestor instanceof IfStatement;
  }

  private static boolean isDjangoView(FunctionDef functionDef) {
    FunctionSymbol functionSymbol = ((FunctionDefImpl) functionDef).functionSymbol();
    return Optional.ofNullable(functionSymbol)
      .map(FunctionSymbolImpl.class::cast)
      .filter(FunctionSymbolImpl::isDjangoView)
      .isPresent();
  }
}
