/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.hotspots;

import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.Tree;

@Rule(key = "S1523")
public class DynamicCodeExecutionCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Make sure that this dynamic injection or execution of code is safe.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      PyCallExpressionTree callExpr = (PyCallExpressionTree) ctx.syntaxNode();
      if (isFuncNameExecOrEval(callExpr)) {
        ctx.addIssue(callExpr, MESSAGE);
      }
    });
  }

  private static boolean isFuncNameExecOrEval(PyCallExpressionTree call) {
    PyExpressionTree expr = call.callee();
    if (expr.is(Tree.Kind.NAME)) {
      String functionName = ((PyNameTree) expr).name();
      return functionName.equals("exec") || functionName.equals("eval");
    }
    return false;
  }
}
