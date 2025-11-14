/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.AwsLambdaChecksUtils;

@Rule(key = "S7614")
public class AsyncAwsLambdaHandlerCheck extends PythonSubscriptionCheck {
  
  private static final String MESSAGE = "Remove the `async` keyword from this AWS Lambda handler definition.";
  
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, AsyncAwsLambdaHandlerCheck::checkAsyncLambdaHandler);
  }
  
  private static void checkAsyncLambdaHandler(SubscriptionContext ctx) {
    var functionDef = (FunctionDef) ctx.syntaxNode();

    var asyncToken = functionDef.asyncKeyword();

    if (asyncToken != null && AwsLambdaChecksUtils.isOnlyLambdaHandler(ctx, functionDef)) {
      ctx.addIssue(asyncToken, MESSAGE);
    }
  }
}
