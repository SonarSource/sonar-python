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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.CfgUtils;
import org.sonar.python.cfg.fixpoint.DefinedVariablesAnalysis;
import org.sonar.python.cfg.fixpoint.DefinedVariablesAnalysis.DefinedVariables;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S3827")
public class ReferencedBeforeAssignmentCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {

    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      List<Symbol> ignoredSymbols = new ArrayList<>();
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (TreeUtils.hasDescendant(functionDef, tree -> tree.is(Tree.Kind.TRY_STMT))) {
        return;
      }
      ControlFlowGraph cfg = ControlFlowGraph.build(functionDef, ctx.pythonFile());
      if (cfg == null) {
        return;
      }
      DefinedVariablesAnalysis analysis = DefinedVariablesAnalysis.analyze(cfg, functionDef.localVariables());
      Set<CfgBlock> unreachableBlocks = CfgUtils.unreachableBlocks(cfg);
      cfg.blocks().forEach(block -> checkCfgBlock(block, ctx, analysis.getDefinedVariables(block), unreachableBlocks, analysis, ignoredSymbols));
    });
  }

  private static void checkCfgBlock(CfgBlock cfgBlock, SubscriptionContext ctx, DefinedVariables definedVariables,
                                    Set<CfgBlock> unreachableBlocks, DefinedVariablesAnalysis analysis, List<Symbol> ignoredSymbols) {
    Map<Symbol, DefinedVariablesAnalysis.VariableDefinition> currentState = new HashMap<>(definedVariables.getIn());
    for (Tree element : cfgBlock.elements()) {
      definedVariables.getSymbolReadWrites(element).forEach((symbol, symbolReadWrite) -> {
        if (symbolReadWrite.isWrite()) {
          currentState.put(symbol, DefinedVariablesAnalysis.VariableDefinition.DEFINED);
        }
        DefinedVariablesAnalysis.VariableDefinition varDef = currentState.getOrDefault(symbol, DefinedVariablesAnalysis.VariableDefinition.DEFINED);
        if (symbolReadWrite.isRead() && isUndefined(varDef)
          && !isSymbolUsedInUnreachableBlocks(analysis, unreachableBlocks, symbol)
          && !isParameter(element)
          && !isTypeAliasStatement(element)
          && !ignoredSymbols.contains(symbol)) {
          ignoredSymbols.add(symbol);
          Usage suspectUsage = symbolReadWrite.usages().get(0);
          PreciseIssue issue = ctx.addIssue(suspectUsage.tree(), symbol.name() + " is used before it is defined. Move the definition before.");
          symbol.usages().stream().filter(u -> !u.equals(suspectUsage)).forEach(us -> issue.secondary(us.tree(), null));
        }
      });
    }
  }

  private static boolean isParameter(Tree element) {
    return element.is(Tree.Kind.PARAMETER);
  }

  private static boolean isTypeAliasStatement(Tree element) {
    return element.is(Tree.Kind.TYPE_ALIAS_STMT);
  }

  private static boolean isSymbolUsedInUnreachableBlocks(DefinedVariablesAnalysis analysis, Set<CfgBlock> unreachableBlocks, Symbol symbol) {
    return unreachableBlocks.stream().anyMatch(b -> analysis.getDefinedVariables(b).isSymbolUsedInBlock(symbol));
  }

  private static boolean isUndefined(DefinedVariablesAnalysis.VariableDefinition varDef) {
    return varDef == DefinedVariablesAnalysis.VariableDefinition.UNDEFINED;
  }
}
