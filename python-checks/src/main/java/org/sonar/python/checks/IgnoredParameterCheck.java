/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.CfgUtils;
import org.sonar.python.cfg.fixpoint.LiveVariablesAnalysis;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;

import static org.sonar.python.checks.DeadStoreUtils.isParameter;
import static org.sonar.python.checks.DeadStoreUtils.isUsedInSubFunction;

@Rule(key = "S1226")
public class IgnoredParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TEMPLATE = "Introduce a new variable or use its initial value before reassigning '%s'.";

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
        List<DeadStoreUtils.UnnecessaryAssignment> unnecessaryAssignments =
          DeadStoreUtils.findUnnecessaryAssignments(block, lva.getLiveVariables(block), functionDef);
        unnecessaryAssignments.stream()
          .filter(assignment -> !assignment.symbol.name().equals("_"))
          .filter((assignment -> isParameter(assignment.element)))
          // symbols should have at least two binding usages
          .filter(assignment -> assignment.symbol.usages().stream().filter(Usage::isBindingUsage).count() > 1)
          // no usages in unreachable blocks
          .filter(assignment -> !isSymbolUsedInUnreachableBlocks(lva, unreachableBlocks, assignment.symbol))
          .filter((assignment -> !isUsedInSubFunction(assignment.symbol, functionDef)))
          .forEach(assignment -> ctx.addIssue(assignment.element, String.format(MESSAGE_TEMPLATE, assignment.symbol.name())));
      });
    });
  }

  private static boolean isSymbolUsedInUnreachableBlocks(LiveVariablesAnalysis lva, Set<CfgBlock> unreachableBlocks, Symbol symbol) {
    return unreachableBlocks.stream().anyMatch(b -> lva.getLiveVariables(b).isSymbolUsedInBlock(symbol));
  }
}
