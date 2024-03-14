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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6929")
public class TfSpecifyReductionAxisCheck extends PythonSubscriptionCheck {

  private static final Set<String> REDUCTION_FUNCTIONS = new HashSet<>(Arrays.asList("reduce_all", "reduce_mean", "reduce_any", "reduce_euclidean_norm", "reduce_logsumexp",
    "reduce_max", "reduce_min", "reduce_prod", "reduce_std", "reduce_sum", "reduce_variance"));
  private static final Set<String> REDUCTION_FUNCTIONS_FQN = new HashSet<>();
  private static final String MESSAGE = "Provide a value for the axis argument.";
  public static final String AXIS_PARAMETER = "axis";
  public static final int AXIS_PARAMETER_POSITION = 1;

  static {
    for (String reductionFunction : REDUCTION_FUNCTIONS) {
      REDUCTION_FUNCTIONS_FQN.add("tensorflow.math." + reductionFunction);
      REDUCTION_FUNCTIONS_FQN.add("tensorflow.tf." + reductionFunction);
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TfSpecifyReductionAxisCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol symbol = callExpression.calleeSymbol();
    if (symbol == null
      || !REDUCTION_FUNCTIONS_FQN.contains(symbol.fullyQualifiedName())
      || TreeUtils.nthArgumentOrKeyword(AXIS_PARAMETER_POSITION, AXIS_PARAMETER, callExpression.arguments()) != null) {
      return;
    }
    context.addIssue(callExpression.callee(), MESSAGE);
  }
}
