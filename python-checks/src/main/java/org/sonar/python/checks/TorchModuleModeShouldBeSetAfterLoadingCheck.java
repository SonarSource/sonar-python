/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6982")
public class TorchModuleModeShouldBeSetAfterLoadingCheck extends PythonSubscriptionCheck {
  private static final Set<String> STATE_SETTING_FUNCTION_FQNS = Set.of("eval", "train");
  private static final String LOAD_STATE_DICT_NAME = "load_state_dict";
  private static final String MESSAGE = "Set the module in training or evaluation mode.";


  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpr = (CallExpression) ctx.syntaxNode();
      List<Usage> receiverUsages = getForwardUsagesOfReceiver(callExpr);
      if (isLoadStateDictCall(callExpr) && !hasEvalOrTrainUsage(receiverUsages) && !isModelPassedOn(receiverUsages)) {
        ctx.addIssue(callExpr.callee(), MESSAGE);
      }
    });
  }

  private static boolean isLoadStateDictCall(CallExpression callExpr) {
    // To properly check if the correct load_state_dict is called, typeshed type information would be required.
    // Since this is currently not possible, we check if the parameter to load_state_dict is torch.load(...),
    // with the assumption that if torch.load is passed to this load_state_dict, it is probably the correct method
    if (callExpr.callee() instanceof QualifiedExpression qualifiedExpr) {
      InferredType qualifierType = qualifiedExpr.qualifier().type();
      boolean isModule = qualifierType.mustBeOrExtend("torch.nn.modules.module.Module")
        || qualifierType.mustBeOrExtend("torch.nn.Module");
      return isModule && LOAD_STATE_DICT_NAME.equals(qualifiedExpr.name().name());
    }
    return false;
  }

  private static List<Usage> getForwardUsagesOfReceiver(CallExpression callExpr) {
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
