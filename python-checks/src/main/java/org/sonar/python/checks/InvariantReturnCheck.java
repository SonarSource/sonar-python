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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.cfg.PythonCfgBranchingBlock;
import org.sonar.python.tree.TreeUtils;

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
      if (cfg != null) {
        List<LatestExecutedBlock> latestExecutedBlocks = collectLatestExecutedBlocks(cfg);
        boolean allBlocksHaveReturnStatement = latestExecutedBlocks.stream().allMatch(LatestExecutedBlock::hasReturnStatement);
        if (latestExecutedBlocks.size() <= 1 || !allBlocksHaveReturnStatement) {
          return;
        }
        boolean noEmptyReturn = latestExecutedBlocks.stream().noneMatch(LatestExecutedBlock::returnNone);
        if (noEmptyReturn && returnExpressionsHaveTheSameValue(latestExecutedBlocks)) {
          PreciseIssue issue = ctx.addIssue(functionDef.name(), MESSAGE);
          latestExecutedBlocks.forEach(block -> issue.secondary(block.returnStatement, "returned value"));
        }
      }
    });
  }

  private static List<LatestExecutedBlock> collectLatestExecutedBlocks(ControlFlowGraph cfg) {
    List<LatestExecutedBlock> collectedBlocks = new ArrayList<>();
    for (CfgBlock predecessor : cfg.end().predecessors()) {
      if (predecessor instanceof PythonCfgBranchingBlock pythonCfgBranchingBlock) {
        collectBranchingBlock(collectedBlocks, pythonCfgBranchingBlock);
      } else if (!endsWithElementKind(predecessor, Kind.RAISE_STMT)) {
        collectedBlocks.add(new LatestExecutedBlock(predecessor));
      }
    }
    return collectedBlocks;
  }

  private static void collectBranchingBlock(List<LatestExecutedBlock> collectedBlocks, PythonCfgBranchingBlock branchingBlock) {
    Tree branchingTree = branchingBlock.branchingTree();
    if (branchingTree.is(Kind.TRY_STMT)) {
      TryStatement tryStatement = (TryStatement) branchingTree;
      if (!TreeUtils.hasDescendant(tryStatement.body(), t -> t.is(Kind.RETURN_STMT))) {
        collectedBlocks.add(new LatestExecutedBlock(branchingBlock));
      }
    } else if (branchingTree.is(Kind.IF_STMT) || branchingTree instanceof Pattern) {
      collectedBlocks.add(new LatestExecutedBlock(branchingBlock));
    } else {
      collectBlocksHavingReturnBeforeExceptOrFinallyBlock(collectedBlocks, branchingBlock);
    }
  }

  private static void collectBlocksHavingReturnBeforeExceptOrFinallyBlock(List<LatestExecutedBlock> collectedBlocks, PythonCfgBranchingBlock branchingBlock) {
    if (branchingBlock.branchingTree().is(Kind.EXCEPT_CLAUSE, Kind.FINALLY_CLAUSE)) {
      for (CfgBlock predecessor : branchingBlock.predecessors()) {
        if (predecessor instanceof PythonCfgBranchingBlock pythonCfgBranchingBlock) {
          collectBlocksHavingReturnBeforeExceptOrFinallyBlock(collectedBlocks, pythonCfgBranchingBlock);
        } else if (endsWithElementKind(predecessor, Kind.RETURN_STMT)) {
          collectedBlocks.add(new LatestExecutedBlock(predecessor));
        }
      }
    }
  }

  private static boolean returnExpressionsHaveTheSameValue(List<LatestExecutedBlock> latestExecutedBlocks) {
    for (int i = 1; i < latestExecutedBlocks.size(); i++) {
      if (!haveTheSameValue(latestExecutedBlocks.get(i - 1), latestExecutedBlocks.get(i))) {
        return false;
      }
    }
    return true;
  }

  private static boolean haveTheSameValue(LatestExecutedBlock left, LatestExecutedBlock right) {
    return haveTheSameValue(left, left.returnExpressions(), right, right.returnExpressions());
  }

  private static boolean haveTheSameValue(LatestExecutedBlock leftBlock, List<? extends Tree> left, LatestExecutedBlock rightBlock, List<? extends Tree> right) {
    if (left.size() != right.size()) {
      return false;
    }
    for (int i = 0; i < left.size(); i++) {
      if (!haveTheSameValue(leftBlock, left.get(i), rightBlock, right.get(i))) {
        return false;
      }
    }
    return true;
  }

  private static boolean haveTheSameValue(LatestExecutedBlock leftBlock, Tree left, LatestExecutedBlock rightBlock, Tree right) {
    if (left.is(Kind.PARENTHESIZED)) {
      return haveTheSameValue(leftBlock, ((ParenthesizedExpression) left).expression(), rightBlock, right);
    } else if (right.is(Kind.PARENTHESIZED)) {
      return haveTheSameValue(leftBlock, left, rightBlock, ((ParenthesizedExpression) right).expression());
    } else if (left.getKind() != right.getKind()) {
      return false;
    } else if (left.is(UNARY_EXPRESSION_KINDS)) {
      return unaryExpressionsHaveTheSameValue(leftBlock, (UnaryExpression) left, rightBlock, (UnaryExpression) right);
    } else if (left.is(BINARY_EXPRESSION_KINDS)) {
      return binaryExpressionsHaveTheSameValue(leftBlock, (BinaryExpression) left, rightBlock, (BinaryExpression) right);
    } else if (left.is(Kind.STRING_LITERAL)) {
      return haveTheSameValue(leftBlock, left.children(), rightBlock, right.children());
    } else if (left.is(Kind.STRING_ELEMENT)) {
      return ((StringElement) left).value().equals(((StringElement) right).value());
    } else if (left.is(Kind.NUMERIC_LITERAL)) {
      return left.firstToken().value().equals(right.firstToken().value());
    } else if (left.is(Kind.NAME)) {
      return identifierHaveTheSameValue(leftBlock, (Name) left, rightBlock, (Name) right);
    }
    return false;
  }

  private static boolean unaryExpressionsHaveTheSameValue(LatestExecutedBlock leftBlock, UnaryExpression left, LatestExecutedBlock rightBlock, UnaryExpression right) {
    // the caller ensure left.getKind() == right.getKind(), so no need to compare left and right UnaryExpression#operator(), it's redundant
    return haveTheSameValue(leftBlock, left.expression(), rightBlock, right.expression());
  }

  private static boolean binaryExpressionsHaveTheSameValue(LatestExecutedBlock leftBlock, BinaryExpression left, LatestExecutedBlock rightBlock, BinaryExpression right) {
    return left.operator().value().equals(right.operator().value()) &&
      haveTheSameValue(leftBlock, left.leftOperand(), rightBlock, right.leftOperand()) &&
      haveTheSameValue(leftBlock, left.rightOperand(), rightBlock, right.rightOperand());
  }

  private static boolean identifierHaveTheSameValue(LatestExecutedBlock leftBlock, Name left, LatestExecutedBlock rightBlock, Name right) {
    if (left.name().equals("True") || left.name().equals("False")) {
      return right.name().equals(left.name());
    }
    Symbol leftSymbol = left.symbol();
    Symbol rightSymbol = right.symbol();
    if (leftSymbol == null || !leftSymbol.equals(rightSymbol)) {
      return false;
    }
    Tree leftBinding = findUniquePreviousBinding(leftBlock, leftSymbol);
    Tree rightBinding = findUniquePreviousBinding(rightBlock, rightSymbol);
    return leftBinding != null && leftBinding == rightBinding;
  }

  private static Tree findUniquePreviousBinding(LatestExecutedBlock context, Symbol identifier) {
    Set<CfgBlock> pushedBlocks = new HashSet<>();
    Deque<CfgBlock> blockToVisit = new ArrayDeque<>();
    blockToVisit.push(context.block);
    pushedBlocks.add(context.block);
    Set<Tree> bindings = new HashSet<>();
    while (!blockToVisit.isEmpty() && bindings.size() < 2) {
      CfgBlock block = blockToVisit.pop();
      Tree binding = findLastBinding(block.elements(), identifier);
      if (binding != null) {
        bindings.add(binding);
      } else {
        for (CfgBlock predecessor : block.predecessors()) {
          if (pushedBlocks.add(predecessor)) {
            blockToVisit.push(predecessor);
          }
        }
      }
    }
    return bindings.size() == 1 ? bindings.iterator().next() : null;
  }

  @Nullable
  private static Tree findLastBinding(List<Tree> elements, Symbol identifier) {
    for (int i = elements.size() - 1; i >= 0; i--) {
      Tree binding = findLastBinding(elements.get(i), identifier);
      if (binding != null) {
        return binding;
      }
    }
    return null;
  }

  @Nullable
  private static Tree findLastBinding(Tree context, Symbol identifier) {
    if (context.is(Kind.NAME)) {
      Name name = (Name) context;
      if (identifier.equals(name.symbol()) && couldBeModified(name)) {
        return name;
      }
    }
    return findLastBinding(context.children(), identifier);
  }

  private static boolean couldBeModified(Name name) {
    Tree child = name;
    Tree parent = child.parent();
    while (parent.is(Kind.STATEMENT_LIST, Kind.PARENTHESIZED, Kind.QUALIFIED_EXPR, Kind.SUBSCRIPTION, Kind.TUPLE)) {
      child = parent;
      parent = child.parent();
    }
    if (parent.is(Kind.RETURN_STMT, Kind.CONDITIONAL_EXPR, Kind.ELSE_CLAUSE, Kind.WHILE_STMT, Kind.EXPRESSION_STMT, Kind.IF_STMT) ||
      parent.is(UNARY_EXPRESSION_KINDS) || parent.is(BINARY_EXPRESSION_KINDS)) {
      return false;
    } else if (parent.is(Kind.ASSIGNMENT_STMT)) {
      return ((AssignmentStatement) parent).lhsExpressions() == child;
    } else if (parent.is(Kind.FOR_STMT)) {
      return ((ForStatement) parent).expressions().contains(child);
    }
    return true;
  }

  private static boolean endsWithElementKind(CfgBlock block, Kind kind) {
    return lastElement(block, kind) != null;
  }

  @Nullable
  private static Tree lastElement(CfgBlock block, Kind kind) {
    List<Tree> elements = block.elements();
    int last = elements.size() - 1;
    return !elements.isEmpty() && elements.get(last).is(kind) ? elements.get(last) : null;
  }

  private static class LatestExecutedBlock {

    private final CfgBlock block;
    @Nullable
    private final ReturnStatement returnStatement;

    private LatestExecutedBlock(CfgBlock block) {
      this.block = block;
      returnStatement = (ReturnStatement) lastElement(block, Kind.RETURN_STMT);
    }

    private boolean hasReturnStatement() {
      return returnStatement != null;
    }

    private List<Expression> returnExpressions() {
      return returnStatement.expressions();
    }

    private boolean returnNone() {
      List<Expression> expressions = returnStatement.expressions();
      return expressions.isEmpty() || (expressions.size() == 1 && expressions.get(0).is(Kind.NONE));
    }

  }

}
