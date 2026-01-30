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
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8412")
public class FastAPIGenericRouteDecoratorCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE =
    "Replace this generic \"route()\" decorator with a specific HTTP method decorator.";

  private static final TypeMatcher ROUTE_DECORATOR_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("fastapi.applications.FastAPI.route"),
    TypeMatchers.isType("fastapi.routing.APIRouter.route")
  );

  private static final Set<String> SINGLE_HTTP_METHODS = Set.of(
    "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD", "TRACE",
    "get", "post", "put", "delete", "patch", "options", "head", "trace"
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF,
      FastAPIGenericRouteDecoratorCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    for (Decorator decorator : functionDef.decorators()) {
      checkDecorator(ctx, decorator);
    }
  }

  private static void checkDecorator(SubscriptionContext ctx, Decorator decorator) {
    CallExpression callExpr = getDecoratorCallExpression(decorator);
    if (callExpr == null) {
      return;
    }

    if (!ROUTE_DECORATOR_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      return;
    }

    RegularArgument methodsArg = TreeUtils.argumentByKeyword("methods", callExpr.arguments());
    if (methodsArg == null) {
      return;
    }

    Expression methodsExpr = methodsArg.expression();
    if (methodsExpr instanceof ListLiteral listLiteral && isSingleHttpMethod(listLiteral)) {
      ctx.addIssue(callExpr.callee(), MESSAGE);
    }
  }

  private static boolean isSingleHttpMethod(ListLiteral listLiteral) {
    List<Expression> elements = listLiteral.elements().expressions();

    if (elements.size() != 1) {
      return false;
    }

    Expression element = elements.get(0);
    StringLiteral stringLiteral = Expressions.extractStringLiteral(element);
    if (stringLiteral == null) {
      return false;
    }

    String methodName = stringLiteral.trimmedQuotesValue();
    return SINGLE_HTTP_METHODS.contains(methodName);
  }

  @CheckForNull
  private static CallExpression getDecoratorCallExpression(Decorator decorator) {
    Expression decoratorExpr = decorator.expression();
    return decoratorExpr instanceof CallExpression callExpr ? callExpr : null;
  }
}
