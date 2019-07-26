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

import com.intellij.lang.ASTNode;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyFunction;
import com.jetbrains.python.psi.PyParameterList;
import org.sonar.check.Rule;
import org.sonar.python.IssueLocation;
import org.sonar.python.PythonCheck;
import org.sonar.python.SubscriptionContext;

@Rule(key = "S2733")
public class ExitHasBadArgumentsCheck extends PythonCheck {

  private static final int EXIT_ARGUMENTS_NUMBER = 4;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.FUNCTION_DECLARATION, ctx -> {
      PyFunction function = (PyFunction) ctx.syntaxNode();
      ASTNode nameNode = function.getNameNode();
      if (nameNode == null || !"__exit__".equals(nameNode.getText())) {
        return;
      }

      PyParameterList parameters = function.getParameterList();
      if (parameters.hasPositionalContainer() || parameters.hasKeywordContainer()) {
        return;
      }

      int numberOfParameters = parameters.getParameters().length;
      if (numberOfParameters < EXIT_ARGUMENTS_NUMBER) {
        raiseIssue(ctx, nameNode, parameters, "Add the missing argument.");
      } else if (numberOfParameters > EXIT_ARGUMENTS_NUMBER) {
        raiseIssue(ctx, nameNode, parameters, "Remove the unnecessary argument.");
      }
    });
  }

  private static void raiseIssue(SubscriptionContext ctx, ASTNode nameNode, PyParameterList parameterList, String message) {
    ctx.addIssue(IssueLocation.preciseLocation(nameNode.getPsi(), parameterList, message));
  }
}

