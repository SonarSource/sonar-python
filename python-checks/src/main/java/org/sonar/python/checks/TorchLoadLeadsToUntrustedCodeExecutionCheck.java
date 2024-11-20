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

import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6985")
public class TorchLoadLeadsToUntrustedCodeExecutionCheck extends PythonSubscriptionCheck {

  public static final String TORCH_LOAD = "torch.load";
  public static final String MESSAGE = "Replace this call with a safe alternative.";
  public static final String PYTHON_FALSE = "False";
  public static final String WEIGHTS_ONLY = "weights_only";

  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> reachingDefinitionsAnalysis =
      new ReachingDefinitionsAnalysis(ctx.pythonFile()));

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && TORCH_LOAD.equals(calleeSymbol.fullyQualifiedName())
        && isWeightsOnlyNotFoundOrSetToFalse(callExpression.arguments())) {

        ctx.addIssue(callExpression.callee(), MESSAGE);
      }
    });
  }

  private boolean isWeightsOnlyNotFoundOrSetToFalse(List<Argument> arguments) {
    RegularArgument weightsOnlyArg = TreeUtils.argumentByKeyword(WEIGHTS_ONLY, arguments);
    if (weightsOnlyArg == null) return !Expressions.containsSpreadOperator(arguments);
    if (weightsOnlyArg.expression() instanceof Name name) {
      return PYTHON_FALSE.equals(name.name()) || isNameSetToFalse(name);
    }
    return false;
  }

  private boolean isNameSetToFalse(Name name) {
    Set<Expression> values = reachingDefinitionsAnalysis.valuesAtLocation(name);
    return values.size() == 1 && values.stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(Name.class))
      .map(Name::name).allMatch(PYTHON_FALSE::equals);
  }


}
