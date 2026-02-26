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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6965")
public class FlaskRouteMethodsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Specify the HTTP methods this route should accept.";
  private static final String FLASK_ROUTE_FQN = "flask.app.Flask.route";
  private static final String BLUEPRINT_ROUTE_FQN = "flask.blueprints.Blueprint.route";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FlaskRouteMethodsCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    functionDef.decorators().stream()
      .filter(decorator -> isFlaskRouteDecoratorWithoutMethods(decorator, ctx))
      .forEach(decorator -> ctx.addIssue(decorator, MESSAGE));
  }

  private static boolean isFlaskRouteDecoratorWithoutMethods(Decorator decorator, SubscriptionContext ctx) {
    Expression expression = decorator.expression();
    if (!(expression instanceof CallExpression callExpr)) {
      return false;
    }

    if (!isFlaskRouteDecorator(callExpr, ctx)) {
      return false;
    }

    return !hasMethodsParameter(callExpr);
  }

  private static boolean isFlaskRouteDecorator(CallExpression callExpr, SubscriptionContext ctx) {
    return TypeMatchers.any(
      TypeMatchers.isType(FLASK_ROUTE_FQN),
      TypeMatchers.isType(BLUEPRINT_ROUTE_FQN)
    ).isTrueFor(callExpr.callee(), ctx);
  }

  private static boolean hasMethodsParameter(CallExpression callExpr) {
    // Check for explicit keyword argument: methods=...
    if (TreeUtils.argumentByKeyword("methods", callExpr.arguments()) != null) {
      return true;
    }

    return callExpr.arguments().stream().anyMatch(TreeUtils::isDoubleStarExpression);
  }
}
