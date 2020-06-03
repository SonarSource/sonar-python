/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.CfgUtils;
import org.sonar.python.cfg.fixpoint.DefinedVariablesAnalysis;
import org.sonar.python.cfg.fixpoint.DefinedVariablesAnalysis.DefinedVariables;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S3827")
public class UndeclaredNameUsageCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      FileInput fileInput = (FileInput) ctx.syntaxNode();
      if (importsManipulatedAllProperty(fileInput)) {
        return;
      }
      UnresolvedSymbolsVisitor unresolvedSymbolsVisitor = new UnresolvedSymbolsVisitor();
      fileInput.accept(unresolvedSymbolsVisitor);
      if (!unresolvedSymbolsVisitor.callGlobalsOrLocals && !unresolvedSymbolsVisitor.hasUnresolvedWildcardImport) {
        addNameIssues(unresolvedSymbolsVisitor.nameIssues, ctx);
      }
    });

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

  private static boolean importsManipulatedAllProperty(FileInput fileInput) {
    return fileInput.globalVariables().stream().anyMatch(s -> s.name().equals("__all__") &&
      s.usages().stream().anyMatch(usage -> usage.kind() == Usage.Kind.IMPORT));
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
        if (symbolReadWrite.isRead() && isUndefined(varDef) && !isSymbolUsedInUnreachableBlocks(analysis, unreachableBlocks, symbol) && !isParameter(element)
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

  private static boolean isSymbolUsedInUnreachableBlocks(DefinedVariablesAnalysis analysis, Set<CfgBlock> unreachableBlocks, Symbol symbol) {
    return unreachableBlocks.stream().anyMatch(b -> analysis.getDefinedVariables(b).isSymbolUsedInBlock(symbol));
  }

  private static boolean isUndefined(DefinedVariablesAnalysis.VariableDefinition varDef) {
    return varDef == DefinedVariablesAnalysis.VariableDefinition.UNDEFINED;
  }

  private static void addNameIssues(Map<String, List<Name>> nameIssues, SubscriptionContext subscriptionContext) {
    nameIssues.forEach((name, list) -> {
      Name first = list.get(0);
      PreciseIssue issue = subscriptionContext.addIssue(first, first.name() + " is not defined. Change its name or define it before using it");
      list.stream().skip(1).forEach(n -> issue.secondary(n, null));
    });
  }

  private static class UnresolvedSymbolsVisitor extends BaseTreeVisitor {

    private boolean hasUnresolvedWildcardImport = false;
    private boolean callGlobalsOrLocals = false;
    private Map<String, List<Name>> nameIssues = new HashMap<>();

    @Override
    public void visitName(Name name) {
      if (name.isVariable() && name.symbol() == null) {
        nameIssues.computeIfAbsent(name.name(), k -> new ArrayList<>()).add(name);
      }
    }

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      hasUnresolvedWildcardImport |= importFrom.hasUnresolvedWildcardImport();
      super.visitImportFrom(importFrom);
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (callExpression.callee().is(Tree.Kind.NAME)) {
        String name = ((Name) callExpression.callee()).name();
        callGlobalsOrLocals |= name.equals("globals") || name.equals("locals");
      }
      super.visitCallExpression(callExpression);
    }
  }
}
