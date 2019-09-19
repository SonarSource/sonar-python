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

import com.sonar.sslr.api.Token;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyParameterListTree;
import org.sonar.python.api.tree.PyParameterTree;
import org.sonar.python.api.tree.Tree;

@Rule(key = "S2733")
public class ExitHasBadArgumentsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_ADD = "Add the missing argument.";
  private static final String MESSAGE_REMOVE = "Remove the unnecessary argument.";

  private static final int EXIT_ARGUMENTS_NUMBER = 4;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      PyFunctionDefTree funcDef = (PyFunctionDefTree) ctx.syntaxNode();
      if (!funcDef.name().name().equals("__exit__")) {
        return;
      }
      PyParameterListTree parameters = funcDef.parameters();
      int arity = 0;
      if(parameters != null) {
        if(parameters.nonTuple().stream().anyMatch(ExitHasBadArgumentsCheck::isStarredParam)) {
          return;
        }
        arity = parameters.all().size();
      }
      raiseIssue(ctx, funcDef, arity);
    });
  }

  private static boolean isStarredParam(PyParameterTree param) {
    return param.starToken() != null;
  }

  private static void raiseIssue(SubscriptionContext ctx, PyFunctionDefTree tree, int argumentsNumber) {
    if (argumentsNumber != EXIT_ARGUMENTS_NUMBER){
      String message = MESSAGE_ADD;
      if (argumentsNumber > EXIT_ARGUMENTS_NUMBER){
        message = MESSAGE_REMOVE;
      }
      Tree funcName = tree.name();
      Token rightParenthesis = tree.rightPar();
      ctx.addIssue(funcName.firstToken(), rightParenthesis, message);
    }
  }
}

