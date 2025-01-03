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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.utils.Expressions.getAssignedName;

@Rule(key = "S6969")
public class SklearnPipelineSpecifyMemoryArgumentCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Specify a memory argument for the pipeline.";
  public static final String MESSAGE_QUICKFIX = "Add the memory argument";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, SklearnPipelineSpecifyMemoryArgumentCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext subscriptionContext) {
    Optional.of(subscriptionContext.syntaxNode())
      .map(CallExpression.class::cast)
      .filter(SklearnPipelineSpecifyMemoryArgumentCheck::isPipelineCreation)
      .ifPresent(
        callExpression -> {
          var memoryArgument = TreeUtils.argumentByKeyword("memory", callExpression.arguments());

          if (memoryArgument != null) {
            return;
          }

          boolean isUsedInAnotherPipeline = getAssignedName(callExpression)
            .map(SklearnPipelineSpecifyMemoryArgumentCheck::isUsedInAnotherPipeline)
            .orElse(false);

          if (isUsedInAnotherPipeline) {
            return;
          }

          createIssue(subscriptionContext, callExpression);
        });
  }

  private static void createIssue(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    var issue = subscriptionContext.addIssue(callExpression.callee(), MESSAGE);
    var quickFix = PythonQuickFix.newQuickFix(MESSAGE_QUICKFIX)
      .addTextEdit(TextEditUtils.insertBefore(callExpression.rightPar(), ", memory=None"))
      .build();
    issue.addQuickFix(quickFix);
  }

  private static boolean isUsedInAnotherPipeline(Name name) {
    Symbol symbol = name.symbol();
    return symbol != null && symbol.usages().stream().filter(usage -> !usage.isBindingUsage()).anyMatch(u -> {
      Tree tree = u.tree();
      CallExpression callExpression = (CallExpression) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.CALL_EXPR);
      while (callExpression != null) {
        if (isUsedBySklearnComposeEstimatorOrPipelineCreation(callExpression)) {
          return true;
        }
        callExpression = (CallExpression) TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.CALL_EXPR);
      }
      return false;
    });
  }

  private static boolean isPipelineCreation(CallExpression callExpression) {
    return Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .map(SklearnPipelineSpecifyMemoryArgumentCheck::isFullyQualifiedNameAPipelineCreation)
      .orElse(false);
  }

  private static boolean isUsedBySklearnComposeEstimatorOrPipelineCreation(CallExpression callExpression) {
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if(calleeSymbol == null) return false;

    String fqn = calleeSymbol.fullyQualifiedName();
    if(fqn == null) return false;

    return isFullyQualifiedNameAPipelineCreation(fqn) || isFullyQualifiedNameASklearnComposeEstimator(fqn);
  }

  private static boolean isFullyQualifiedNameAPipelineCreation(String fqn) {
    return "sklearn.pipeline.Pipeline".equals(fqn) || "sklearn.pipeline.make_pipeline".equals(fqn);
  }

  private static boolean isFullyQualifiedNameASklearnComposeEstimator(String fqn) {
    return fqn.startsWith("sklearn.compose.");
  }

}
