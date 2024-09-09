/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6983")
public class PyTorchDataLoaderNumWorkersCheck extends PythonSubscriptionCheck {
  private static final String TORCH_UTILS_DATA_DATA_LOADER = "torch.utils.data.DataLoader";
  public static final String MESSAGE = "Specify the `num_workers` parameter.";
  public static final String NUM_WORKERS_ARG_NAME = "num_workers";
  public static final int NUM_WORKERS_ARG_POSITION = 5;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      List<Argument> arguments = callExpression.arguments();
      if (calleeSymbol != null && TORCH_UTILS_DATA_DATA_LOADER.equals(calleeSymbol.fullyQualifiedName())
        && checkIfNumWorkersArgIsPresent(arguments)
        && !checkIfUnpackArgIsPresent(arguments)) {

        ctx.addIssue(callExpression.callee(), MESSAGE);
      }
    });
  }

  private static boolean checkIfNumWorkersArgIsPresent(List<Argument> arguments) {
    RegularArgument numWorkersArg = TreeUtils.nthArgumentOrKeyword(NUM_WORKERS_ARG_POSITION, NUM_WORKERS_ARG_NAME, arguments);
    return numWorkersArg == null;
  }

  private static boolean checkIfUnpackArgIsPresent(List<Argument> arguments) {
    return arguments.stream().anyMatch(UnpackingExpression.class::isInstance);
  }
}
