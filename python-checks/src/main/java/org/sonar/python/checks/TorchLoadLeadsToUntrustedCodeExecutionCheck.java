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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6985")
public class TorchLoadLeadsToUntrustedCodeExecutionCheck extends PythonSubscriptionCheck {

  public static final String TORCH_LOAD = "torch.load";
  public static final String MESSAGE = "Replace this call with a safe alternative";
  public static final String PYTHON_FALSE = "False";
  public static final String WEIGHTS_ONLY = "weights_only";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && TORCH_LOAD.equals(calleeSymbol.fullyQualifiedName()) && !isWeightsOnlySetToTrue(callExpression.arguments())) {
        ctx.addIssue(callExpression.callee(), MESSAGE);
      }
    });
  }

  private static boolean isWeightsOnlySetToTrue(List<Argument> arguments) {
    RegularArgument weightsOnlyArg = TreeUtils.argumentByKeyword(WEIGHTS_ONLY, arguments);
    if (weightsOnlyArg != null) {
      Expression weightsOnlyArgExpr = weightsOnlyArg.expression();
      return weightsOnlyArgExpr.is(Tree.Kind.NAME) && !PYTHON_FALSE.equals(((Name) weightsOnlyArgExpr).name());
    }
    return false;
  }


}
