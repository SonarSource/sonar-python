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

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6982")
public class TorchModuleModeShouldBeSetAfterLoadingCheck extends PythonSubscriptionCheck {
  private static final Set<String> STATE_SETTING_FUNCTION_FQNS = Set.of("eval", "train");
  private static final String TORCH_LOAD_FQN = "torch.load";
  private static final String LOAD_STATE_DICT_NAME = "load_state_dict";
  private static final String MESSAGE = "Set the module in training or evaluation mode.";
  private static final int IS_TORCH_LOAD_CALL_MAX_RECURSIVE_COUNTER = 10;

  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> reachingDefinitionsAnalysis =
      new ReachingDefinitionsAnalysis(ctx.pythonFile()));

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpr = (CallExpression) ctx.syntaxNode();
      List<Usage> receiverUsages = getForwardUsages(callExpr);
      if (isLoadStateDictCall(callExpr) && !hasEvalOrTrainUsage(receiverUsages) && !isModelPassedOn(receiverUsages)) {
        ctx.addIssue(callExpr.callee(), MESSAGE);
      }
    });
  }

  private boolean isLoadStateDictCall(CallExpression callExpr) {
    // To properly check if the correct load_state_dict is called, typeshed type information would be required.
    // Since this is currently not possible, we check if the parameter to load_state_dict is torch.load(...),
    // with the assumption that if torch.load is passed to this load_state_dict, it is probably the correct method
    if (callExpr.callee() instanceof QualifiedExpression qualifiedExpr) {
      return qualifiedExpr.qualifier().type().mustBeOrExtend("torch.nn.modules.module.Module")
        && LOAD_STATE_DICT_NAME.equals(qualifiedExpr.name().name()) && containsTorchLoadCall(callExpr.arguments());
    }
    return false;
  }

  private boolean containsTorchLoadCall(List<Argument> args) {
    return args.stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
      .anyMatch(arg -> isTorchLoadCall(arg.expression(), 0));
  }

  private boolean isTorchLoadCall(Expression expr, int recursiveCounter) {
    if (recursiveCounter > IS_TORCH_LOAD_CALL_MAX_RECURSIVE_COUNTER) {
      return false;
    } else if (expr instanceof CallExpression callExpr) {
      Symbol calleeSymbol = callExpr.calleeSymbol();
      return calleeSymbol != null && TORCH_LOAD_FQN.equals(calleeSymbol.fullyQualifiedName());
    } else if (expr instanceof Name name) {
      return reachingDefinitionsAnalysis.valuesAtLocation(name).stream()
        .anyMatch(definitionExpr -> isTorchLoadCall(definitionExpr, recursiveCounter + 1));
    } else {
      return false;
    }
  }

  private static List<Usage> getForwardUsages(CallExpression callExpr) {
    List<Usage> usages = getFunctionCallReceiverName(callExpr)
      .flatMap(name -> Optional.ofNullable(name.symbol()))
      .map(Symbol::usages)
      .orElse(Collections.emptyList());

    return usages.stream()
      .filter(usage -> usage.tree().firstToken().line() > callExpr.firstToken().line())
      .toList();
  }

  private static Optional<Name> getFunctionCallReceiverName(CallExpression callExpr) {
    return Optional.ofNullable(callExpr.callee())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .flatMap(qualifiedExpr -> Optional.ofNullable(qualifiedExpr.qualifier()))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class));
  }

  private static boolean hasEvalOrTrainUsage(List<Usage> usages) {
    return usages.stream().anyMatch(TorchModuleModeShouldBeSetAfterLoadingCheck::isEvalOrTrain);
  }

  private static boolean isEvalOrTrain(Usage usage) {
    Tree callTree = TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.CALL_EXPR);
    if (callTree != null) {
      CallExpression usageCall = (CallExpression) callTree;
      Symbol usageCallSymbol = usageCall.calleeSymbol();
      return usageCallSymbol != null && STATE_SETTING_FUNCTION_FQNS.contains(usageCallSymbol.name());
    }
    return false;
  }

  private static boolean isModelPassedOn(List<Usage> usages) {
    return usages.stream().anyMatch(TorchModuleModeShouldBeSetAfterLoadingCheck::isPassingModel);
  }

  private static boolean isPassingModel(Usage usage) {
    return TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.CALL_EXPR) != null;
  }
}
