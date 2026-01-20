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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S8375")
public class FlaskPreprocessRequestCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Handle the return value of \"preprocess_request()\" to ensure before-request handlers' responses are not ignored.";
  private static final TypeMatcher PREPROCESS_REQUEST_MATCHER = TypeMatchers.isType("flask.app.Flask.preprocess_request");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.EXPRESSION_STMT, FlaskPreprocessRequestCheck::checkExpressionStatement);
  }

  private static void checkExpressionStatement(SubscriptionContext ctx) {
    ExpressionStatement expressionStatement = (ExpressionStatement) ctx.syntaxNode();

    // By only checking expressions(), we filter out uses of prepocess_request where the return value is used.
    for (Expression expression : expressionStatement.expressions()) {
      if (expression.is(Tree.Kind.CALL_EXPR)) {
        checkCallExpression(ctx, (CallExpression) expression);
      }
    }
  }

  private static void checkCallExpression(SubscriptionContext ctx, CallExpression callExpression) {
    if (PREPROCESS_REQUEST_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      ctx.addIssue(callExpression.callee(), MESSAGE);
    }
  }
}

