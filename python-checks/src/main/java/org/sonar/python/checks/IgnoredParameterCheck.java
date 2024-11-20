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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.CfgUtils;
import org.sonar.python.cfg.fixpoint.LiveVariablesAnalysis;
import org.sonar.python.checks.utils.DeadStoreUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.utils.DeadStoreUtils.isParameter;
import static org.sonar.python.checks.utils.DeadStoreUtils.isUsedInSubFunction;

@Rule(key = "S1226")
public class IgnoredParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TEMPLATE = "Introduce a new variable or use its initial value before reassigning '%s'.";
  private static final String SECONDARY_MESSAGE_TEMPLATE = "'%s' is reassigned here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      ControlFlowGraph cfg = ControlFlowGraph.build(functionDef, ctx.pythonFile());
      if (cfg == null) {
        return;
      }
      LiveVariablesAnalysis lva = LiveVariablesAnalysis.analyze(cfg);
      Set<CfgBlock> unreachableBlocks = CfgUtils.unreachableBlocks(cfg);
      cfg.blocks().forEach(block -> {
        var unnecessaryAssignments = DeadStoreUtils.findUnnecessaryAssignments(block, lva.getLiveVariables(block), functionDef);
        unnecessaryAssignments.stream()
          .filter(assignment -> !assignment.symbol.name().equals("_"))
          .filter((assignment -> isParameter(assignment.element)))
          // symbols should have at least two binding usages
          .filter(assignment -> assignment.symbol.usages().stream().filter(Usage::isBindingUsage).count() > 1)
          // no usages in unreachable blocks
          .filter(assignment -> !isSymbolUsedInUnreachableBlocks(lva, unreachableBlocks, assignment.symbol))
          .filter((assignment -> !isUsedInSubFunction(assignment.symbol, functionDef)))
          .forEach(assignment -> {
            var issue = ctx.addIssue(assignment.element, String.format(MESSAGE_TEMPLATE, assignment.symbol.name()));
            assignment.symbol.usages().stream()
              .filter(Usage::isBindingUsage)
              .filter(u -> u.kind() != Usage.Kind.PARAMETER)
              .map(Usage::tree)
              .collect(TreeUtils.groupAssignmentByParentStatementList())
              .values()
              .stream()
              .sorted(TreeUtils.getTreeByPositionComparator())
              .map(IgnoredParameterCheck::mapToParentAssignmentStatementOrExpression)
              .forEach(tree -> issue.secondary(tree, String.format(SECONDARY_MESSAGE_TEMPLATE, assignment.symbol.name())));
          });
      });
    });
  }

  private static Tree mapToParentAssignmentStatementOrExpression(Tree tree) {
    var assignment = TreeUtils.firstAncestor(tree, parent -> parent.is(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ASSIGNMENT_EXPRESSION));
    if (assignment != null) {
      return assignment;
    }
    return tree;
  }

  private static boolean isSymbolUsedInUnreachableBlocks(LiveVariablesAnalysis lva, Set<CfgBlock> unreachableBlocks, Symbol symbol) {
    return unreachableBlocks.stream().anyMatch(b -> lva.getLiveVariables(b).isSymbolUsedInBlock(symbol));
  }
}
