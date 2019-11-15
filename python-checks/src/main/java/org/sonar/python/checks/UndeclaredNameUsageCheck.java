/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
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
  private boolean hasWildcardImport = false;
  private boolean callGlobalsOrLocals = false;

  @Override
  public void initialize(Context context) {

    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      FileInput fileInput = (FileInput) ctx.syntaxNode();
      ExceptionVisitor exceptionVisitor = new ExceptionVisitor();
      fileInput.accept(exceptionVisitor);
      hasWildcardImport = exceptionVisitor.hasWildcardImport;
      callGlobalsOrLocals = exceptionVisitor.callGlobalsOrLocals;
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, ctx -> {
      Name name = (Name) ctx.syntaxNode();
      if (!callGlobalsOrLocals && !hasWildcardImport && name.isVariable() && name.symbol() == null) {
        ctx.addIssue(name, name.name() + " is not defined. Change its name or define it before using it");
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
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
      cfg.blocks().forEach(block -> checkCfgBlock(block, ctx, analysis.getDefinedVariables(block), unreachableBlocks, analysis));
    });
  }

  private static void checkCfgBlock(
    CfgBlock cfgBlock, SubscriptionContext ctx, DefinedVariables definedVariables, Set<CfgBlock> unreachableBlocks, DefinedVariablesAnalysis analysis) {
    Map<Symbol, DefinedVariablesAnalysis.VariableDefinition> currentState = new HashMap<>(definedVariables.getIn());
    for (Tree element : cfgBlock.elements()) {
      definedVariables.getVariableUsages(element).forEach((symbol, symbolUsage) -> {
        if (symbolUsage.isWrite()) {
          currentState.put(symbol, DefinedVariablesAnalysis.VariableDefinition.DEFINED);
        }
        DefinedVariablesAnalysis.VariableDefinition varDef = currentState.getOrDefault(symbol, DefinedVariablesAnalysis.VariableDefinition.DEFINED);
        if (symbolUsage.isRead() && isUndefined(varDef) && !isSymbolUsedInUnreachableBlocks(analysis, unreachableBlocks, symbol) && !isParameter(element)) {
          ctx.addIssue(element, symbol.name() + " is used before it is defined. Move the definition before.");
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

  private static class ExceptionVisitor extends BaseTreeVisitor {
    private boolean hasWildcardImport = false;
    private boolean callGlobalsOrLocals = false;

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      hasWildcardImport |= importFrom.isWildcardImport();
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
