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

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8392")
public class BindToAllNetworkInterfacesCheck extends PythonSubscriptionCheck {

  private static final String ALL_NETWORK_INTERFACES = "0.0.0.0";
  private static final String MESSAGE = "Avoid binding the application to all network interfaces.";
  private static final TypeMatcher UVICORN_APP_RUN_TYPE_MATCHER = TypeMatchers.isType("uvicorn.run");
  private static final TypeMatcher FLASK_APP_RUN_TYPE_MATCHER = TypeMatchers.isType("flask.app.Flask.run");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, BindToAllNetworkInterfacesCheck::checkFunctionCalls);
  }
  private static void checkFunctionCalls(SubscriptionContext ctx) {
    CallExpression callExpr = ((CallExpression) ctx.syntaxNode());
    RegularArgument hostArgument = null;
    if (UVICORN_APP_RUN_TYPE_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      hostArgument = TreeUtils.argumentByKeyword("host", callExpr.arguments());
    } else if (FLASK_APP_RUN_TYPE_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      hostArgument = TreeUtils.nthArgumentOrKeyword(0, "host", callExpr.arguments());
    }
    if (isHostBoundToAll(hostArgument)) {
      ctx.addIssue(callExpr.callee(), MESSAGE);
    }
  }

  private static boolean isHostBoundToAll(@Nullable RegularArgument hostArgument) {
    if (hostArgument==null){
      return false;
    }
    StringLiteral hostValue = Expressions.extractStringLiteral(hostArgument.expression());
    return hostValue != null && ALL_NETWORK_INTERFACES.equals(hostValue.trimmedQuotesValue());
  }
}

