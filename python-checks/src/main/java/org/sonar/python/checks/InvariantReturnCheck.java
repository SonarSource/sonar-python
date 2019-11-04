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
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.cfg.PythonCfgBranchingBlock;

@Rule(key = "S3516")
public class InvariantReturnCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Refactor this method to not always return the same value.";

  private static final Kind[] BINARY_EXPRESSION_KINDS = {
    Kind.PLUS, Kind.MINUS, Kind.MULTIPLICATION, Kind.DIVISION, Kind.FLOOR_DIVISION, Kind.MODULO,
    Kind.MATRIX_MULTIPLICATION, Kind.SHIFT_EXPR, Kind.BITWISE_AND, Kind.BITWISE_OR, Kind.BITWISE_XOR,
    Kind.AND, Kind.OR, Kind.COMPARISON, Kind.POWER};

  private static final Kind[] UNARY_EXPRESSION_KINDS = {
    Kind.UNARY_MINUS, Kind.UNARY_PLUS, Kind.BITWISE_COMPLEMENT, Kind.NOT};

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      ControlFlowGraph cfg = ControlFlowGraph.build(functionDef, ctx.pythonFile());
      if (cfg == null) {
        return;
      }
      List<ReturnStatement> allReturns = collectLatestExecutedBlocks(cfg).stream()
        .map(InvariantReturnCheck::extractReturnStatementFromBlock)
        .collect(Collectors.toList());

      boolean notAllPredecessorsAreReturn = allReturns.stream().anyMatch(Objects::isNull);
      if (allReturns.size() <= 1 || notAllPredecessorsAreReturn) {
        return;
      }
      List<List<Expression>> returnExpressions = allReturns.stream().map(ReturnStatement::expressions).collect(Collectors.toList());
      boolean noneReturnVoid = returnExpressions.stream().noneMatch(List::isEmpty);
      if (noneReturnVoid && returnExpressionsHaveTheSameValue(returnExpressions)) {
        PreciseIssue issue = ctx.addIssue(functionDef.name(), MESSAGE);
        allReturns.forEach(returnStatement -> issue.secondary(returnStatement, "returned value"));
      }
    });
  }

  private static Set<CfgBlock> collectLatestExecutedBlocks(ControlFlowGraph cfg) {
    Set<CfgBlock> predecessors = new HashSet<>();
    for (CfgBlock predecessor : cfg.end().predecessors()) {
      if (predecessor instanceof PythonCfgBranchingBlock) {
        collectBlocksHavingReturnBeforeExceptOrFinallyBlock(predecessors, (PythonCfgBranchingBlock) predecessor);
      } else {
        predecessors.add(predecessor);
      }
    }
    return predecessors;
  }

  private static void collectBlocksHavingReturnBeforeExceptOrFinallyBlock(Set<CfgBlock> predecessors, PythonCfgBranchingBlock branchingBlock) {
    if (branchingBlock.branchingTree().is(Kind.EXCEPT_CLAUSE, Kind.FINALLY_CLAUSE)) {
      for (CfgBlock predecessor : branchingBlock.predecessors()) {
        if (predecessor instanceof PythonCfgBranchingBlock) {
          collectBlocksHavingReturnBeforeExceptOrFinallyBlock(predecessors, (PythonCfgBranchingBlock) predecessor);
        } else if (hasReturnStatement(predecessor)) {
          predecessors.add(predecessor);
        }
      }
    }
  }

  private static boolean returnExpressionsHaveTheSameValue(List<List<Expression>> returnExpressions) {
    for (int i = 1; i < returnExpressions.size(); i++) {
      if (!haveTheSameValue(returnExpressions.get(i - 1), returnExpressions.get(i))) {
        return false;
      }
    }
    return true;
  }

  private static boolean haveTheSameValue(List<Expression> left, List<Expression> right) {
    if (left.size() != right.size()) {
      return false;
    }
    for (int i = 0; i < left.size(); i++) {
      if (!haveTheSameValue(left.get(i), right.get(i))) {
        return false;
      }
    }
    return true;
  }

  private static boolean haveTheSameValue(Expression left, Expression right) {
    return isConstantExpression(left) && CheckUtils.areEquivalent(left, right);
  }

  private static boolean isConstantExpression(Expression expression) {
    if (expression.is(Kind.PARENTHESIZED)) {
      return isConstantExpression(((ParenthesizedExpression) expression).expression());
    } else {
      if (expression.is(BINARY_EXPRESSION_KINDS)) {
        BinaryExpression binaryExpression = (BinaryExpression) expression;
        return isConstantExpression(binaryExpression.leftOperand()) && isConstantExpression(binaryExpression.rightOperand());
      } else if (expression.is(UNARY_EXPRESSION_KINDS)) {
        return isConstantExpression(((UnaryExpression) expression).expression());
      }
    }
    return expression.is(Kind.NONE, Kind.NUMERIC_LITERAL, Kind.STRING_LITERAL);
  }

  private static boolean hasReturnStatement(CfgBlock block) {
    return extractReturnStatementFromBlock(block) != null;
  }

  @Nullable
  private static ReturnStatement extractReturnStatementFromBlock(CfgBlock block) {
    List<Tree> elements = block.elements();
    if (!elements.isEmpty()) {
      Tree last = elements.get(elements.size() - 1);
      if (last.is(Kind.RETURN_STMT)) {
        return (ReturnStatement) last;
      }
    }
    return null;
  }

}
