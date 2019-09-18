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
package org.sonar.python.checks;

import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.TreeSymbol;

@Rule(key = "S1481")
public class UnusedLocalVariableCheck extends PythonSubscriptionCheck {

  private static final Pattern IDENTIFIER_SEPARATOR = Pattern.compile("[^a-zA-Z0-9_]+");

  private static final String MESSAGE = "Remove the unused local variable \"%s\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> {
      PyFunctionDefTree functionTree = (PyFunctionDefTree) ctx.syntaxNode();
      // https://docs.python.org/3/library/functions.html#locals
      if (isCallingLocalsFunction(functionTree)) {
        return;
      }
      for (TreeSymbol symbol : functionTree.localVariables()) {
        if (!"_".equals(symbol.name()) && symbol.usages().size() == 1) {
          symbol.usages().forEach(usage -> ctx.addIssue(usage, String.format(MESSAGE, symbol.name())));
        }
      }
    });
  }

  private static boolean isCallingLocalsFunction(PyFunctionDefTree functionTree) {
    return functionTree
      .descendants(Kind.CALL_EXPR)
      .map(PyCallExpressionTree.class::cast)
      .map(PyCallExpressionTree::callee)
      .anyMatch(callee -> callee.is(Kind.NAME) && "locals".equals(((PyNameTree) callee).name()));
  }
}
