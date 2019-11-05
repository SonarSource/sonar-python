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

import java.util.HashSet;
import java.util.ListIterator;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.cfg.LiveVariablesAnalysis;
import org.sonar.python.semantic.Symbol;
import org.sonar.python.semantic.Usage;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1854")
public class DeadStoreCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TEMPLATE = "Remove this useless assignment to local variable '%s'.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (TreeUtils.hasDescendant(functionDef, tree -> tree.is(Tree.Kind.TRY_STMT))) {
        return;
      }
      ControlFlowGraph cfg = ControlFlowGraph.build(functionDef, ctx.pythonFile());
      if (cfg == null) {
        return;
      }
      LiveVariablesAnalysis lva = LiveVariablesAnalysis.analyze(cfg);
      cfg.blocks().forEach(block -> verifyBlock(ctx, block, lva.getLiveVariables(block), lva.getReadSymbols(), functionDef));
    });
  }

  /**
   * Bottom-up approach, keeping track of which variables will be read by successor elements.
   */
  private static void verifyBlock(SubscriptionContext ctx, CfgBlock block, LiveVariablesAnalysis.LiveVariables blockLiveVariables,
                                  Set<Symbol> readSymbols, FunctionDef functionDef) {

    Set<Symbol> willBeRead = new HashSet<>(blockLiveVariables.getOut());
    ListIterator<Tree> elementsReverseIterator = block.elements().listIterator(block.elements().size());
    while (elementsReverseIterator.hasPrevious()) {
      Tree element = elementsReverseIterator.previous();
      blockLiveVariables.getVariableUsages(element).forEach((symbol, usage) -> {
        if (!readSymbols.contains(symbol)) {
          // will be reported by S1481
          return;
        }
        if (usage.isWrite() && !usage.isRead()) {
          if (!willBeRead.contains(symbol) && functionDef.localVariables().contains(symbol) && !isException(symbol, element, functionDef)) {
            String message = isMultipleAssignement(element)
              ? "Rename \"" + symbol.name() + "\" to \"_\" as it is not used after assignment."
              : String.format(MESSAGE_TEMPLATE, symbol.name());
            ctx.addIssue(element, message);
          }
          willBeRead.remove(symbol);
        } else if (usage.isRead()) {
          willBeRead.add(symbol);
        }
      });
    }
  }

  private static boolean isMultipleAssignement(Tree element) {
    return element.is(Tree.Kind.ASSIGNMENT_STMT) &&
      ((AssignmentStatement) element).lhsExpressions().stream().anyMatch(lhsExpression -> lhsExpression.expressions().size() > 1);
  }

  private static boolean isException(Symbol symbol, Tree element, FunctionDef functionDef) {
    return isAssignmentToFalsyOrTrueLiteral(element)
      || isFunctionDeclarationSymbol(symbol)
      || isLoopDeclarationSymbol(symbol, element)
      || isWithInstance(element)
      || isUsedInSubFunction(symbol, functionDef);
  }

  private static boolean isLoopDeclarationSymbol(Symbol symbol, Tree element) {
    return symbol.usages().stream().anyMatch(u -> u.kind() == Usage.Kind.LOOP_DECLARATION)
      && TreeUtils.firstAncestorOfKind(element, Tree.Kind.FOR_STMT) != null;
  }

  private static boolean isWithInstance(Tree element) {
    return element.is(Tree.Kind.WITH_ITEM);
  }

  private static boolean isAssignmentToFalsyOrTrueLiteral(Tree element) {
    if (element.is(Tree.Kind.ASSIGNMENT_STMT)) {
      Expression assignedValue = ((AssignmentStatement) element).assignedValue();
      return Expressions.isFalsy(assignedValue)
        || (assignedValue.is(Tree.Kind.NAME) && "True".equals(((Name) assignedValue).name()))
        || isNumericLiteralOne(assignedValue)
        || isMinusOne(assignedValue);
    }
    return false;
  }

  private static boolean isMinusOne(Expression assignedValue) {
    if (assignedValue.is(Tree.Kind.UNARY_MINUS)) {
      Expression expression = ((UnaryExpression) assignedValue).expression();
      return isNumericLiteralOne(expression);
    }
    return false;
  }

  private static boolean isNumericLiteralOne(Expression expression) {
    return (expression.is(Tree.Kind.NUMERIC_LITERAL) && "1".equals((((NumericLiteral) expression).valueAsString())));
  }

  private static boolean isFunctionDeclarationSymbol(Symbol symbol) {
    return symbol.usages().stream().anyMatch(u -> u.kind() == Usage.Kind.FUNC_DECLARATION);
  }

  private static boolean isUsedInSubFunction(Symbol symbol, FunctionDef functionDef) {
    return symbol.usages().stream()
      .anyMatch(usage -> TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA) != functionDef);
  }
}
