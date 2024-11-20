/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6929")
public class TfPyTorchSpecifyReductionAxisCheck extends PythonSubscriptionCheck {

  private static final Set<String> TF_REDUCTION_FUNCTIONS = new HashSet<>(Arrays.asList("reduce_all", "reduce_mean", "reduce_any",
    "reduce_euclidean_norm", "reduce_logsumexp",
    "reduce_max", "reduce_min", "reduce_prod", "reduce_std", "reduce_sum", "reduce_variance"));
  private static final Set<String> TF_REDUCTION_FUNCTIONS_FQN = new HashSet<>();
  private static final String TF_MESSAGE = "Provide a value for the axis argument.";
  public static final String AXIS_PARAMETER = "axis";
  public static final int AXIS_PARAMETER_POSITION = 1;

  static {
    for (String reductionFunction : TF_REDUCTION_FUNCTIONS) {
      TF_REDUCTION_FUNCTIONS_FQN.add("tensorflow.math." + reductionFunction);
      TF_REDUCTION_FUNCTIONS_FQN.add("tensorflow.tf." + reductionFunction);
    }
  }

  private static final String PY_TORCH_MESSAGE = "Provide a value for the dim argument.";
  public static final String DIM_PARAMETER = "dim";
  public static final int NO_POSITIONAL_ARG = -1;

  /**
   * Contains a list of reduction functions with a {@code dim} parameter and the position of the dim argument in that function.
   */
  private static final Map<String, Integer> PY_TORCH_REDUCTION_FUNCTIONS_DIM_POS = Map.ofEntries(
    Map.entry("torch.argmin", 1),
    Map.entry("torch.aminmax", NO_POSITIONAL_ARG),
    Map.entry("torch.nanmean", 1),
    Map.entry("torch.mode", 1),
    Map.entry("torch.norm", 2),
    Map.entry("torch.quantile", 2),
    Map.entry("torch.nanquantile", 2),
    Map.entry("torch.std", 1),
    Map.entry("torch.std_mean", 1),
    Map.entry("torch.unique", 4),
    Map.entry("torch.unique_consecutive", 3),
    Map.entry("torch.var", 1),
    Map.entry("torch.var_mean", 1),
    Map.entry("torch.count_nonzero", 1)
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TfPyTorchSpecifyReductionAxisCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol symbol = callExpression.calleeSymbol();
    if (symbol != null && !Expressions.containsSpreadOperator(callExpression.arguments())) {
      if (isTfReductionMissingAxisArg(symbol, callExpression)) {
        PreciseIssue issue = context.addIssue(callExpression.callee(), TF_MESSAGE);
        createTfQuickFix(callExpression).ifPresent(issue::addQuickFix);
      }

      if (isPyTorchReductionMissingDimArg(symbol, callExpression)) {
        context.addIssue(callExpression.callee(), PY_TORCH_MESSAGE);
      }
    }
  }

  private static Optional<PythonQuickFix> createTfQuickFix(CallExpression callExpression) {
    if(callExpression.arguments().isEmpty()) {
      return Optional.empty();
    }
    return Optional.of(PythonQuickFix.newQuickFix("Add axis parameter", TextEditUtils.insertBefore(callExpression.rightPar(), ", axis=None")));
  }

  private static boolean isTfReductionMissingAxisArg(Symbol symbol, CallExpression callExpression) {
    return TF_REDUCTION_FUNCTIONS_FQN.contains(symbol.fullyQualifiedName())
      && TreeUtils.nthArgumentOrKeyword(AXIS_PARAMETER_POSITION, AXIS_PARAMETER, callExpression.arguments()) == null;
  }

  private static boolean isPyTorchReductionMissingDimArg(Symbol symbol, CallExpression callExpression) {
    String fqn = symbol.fullyQualifiedName();
    return fqn != null && PY_TORCH_REDUCTION_FUNCTIONS_DIM_POS.containsKey(fqn)
      && TreeUtils.nthArgumentOrKeyword(PY_TORCH_REDUCTION_FUNCTIONS_DIM_POS.get(fqn), DIM_PARAMETER, callExpression.arguments()) == null;
  }
}
