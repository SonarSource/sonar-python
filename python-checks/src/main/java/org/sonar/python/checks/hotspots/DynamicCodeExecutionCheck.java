/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S1523")
public class DynamicCodeExecutionCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Make sure that this dynamic injection or execution of code is safe.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpr = (CallExpression) ctx.syntaxNode();
      if (isFuncNameExecOrEval(callExpr)) {
        ctx.addIssue(callExpr, MESSAGE);
      }
    });
  }

  private static boolean isFuncNameExecOrEval(CallExpression call) {
    Expression expr = call.callee();
    if (expr.is(Tree.Kind.NAME)) {
      String functionName = ((Name) expr).name();
      return functionName.equals("exec") || functionName.equals("eval");
    }
    return false;
  }
}
