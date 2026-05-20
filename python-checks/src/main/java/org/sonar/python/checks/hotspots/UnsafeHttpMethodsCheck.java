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
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.tree.FunctionDefImpl;

import static org.sonar.plugins.python.api.tree.Tree.Kind.FUNCDEF;

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
    context.registerSyntaxNodeConsumer(FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (isDjangoView(functionDef)) {
        checkDjangoView(functionDef, ctx);
      }
    });
  }

  private static void checkDjangoView(FunctionDef functionDef, SubscriptionContext ctx) {
    for (Decorator decorator : functionDef.decorators()) {
      Expression decoratorExpr = decorator.expression();
      if (COMPLIANT_DECORATORS.isTrueFor(decoratorExpr, ctx) ||
        isCallRequiredHTTPMethod(decoratorExpr, ctx)) {
        return;
      }
      if (!isKnownNonCompliantDjangoHttpDecorator(decoratorExpr)) {
        // Not from django.views.decorators.http — unknown or user-defined decorator
        // may restrict methods, so give benefit of the doubt.
        return;
      }
    }
    ctx.addIssue(functionDef.name(), MESSAGE);
  }

  private static boolean isCallRequiredHTTPMethod(Expression expression, SubscriptionContext ctx) {
    return expression instanceof CallExpression callExpression &&
      IS_REQUIRED_HTTP_METHOD.isTrueFor(callExpression.callee(), ctx);
  }

  private static boolean isKnownNonCompliantDjangoHttpDecorator(Expression expression) {
    var type = expression instanceof CallExpression callExpr
      ? callExpr.callee().typeV2()
      : expression.typeV2();
    return type instanceof UnknownType.UnresolvedImportType uit &&
      uit.importPath().startsWith("django.views.decorators.http.");
  }

  private static boolean isDjangoView(FunctionDef functionDef) {
    FunctionSymbol functionSymbol = ((FunctionDefImpl) functionDef).functionSymbol();
    return Optional.ofNullable(functionSymbol)
      .map(FunctionSymbolImpl.class::cast)
      .filter(FunctionSymbolImpl::isDjangoView)
      .isPresent();
  }
}
