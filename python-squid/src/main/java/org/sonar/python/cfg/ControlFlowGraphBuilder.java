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
package org.sonar.python.cfg;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.python.api.tree.BreakStatement;
import org.sonar.python.api.tree.ContinueStatement;
import org.sonar.python.api.tree.ElseStatement;
import org.sonar.python.api.tree.ExceptClause;
import org.sonar.python.api.tree.FinallyClause;
import org.sonar.python.api.tree.ForStatement;
import org.sonar.python.api.tree.IfStatement;
import org.sonar.python.api.tree.ReturnStatement;
import org.sonar.python.api.tree.Statement;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TryStatement;
import org.sonar.python.api.tree.WhileStatement;

public class ControlFlowGraphBuilder {

  private PythonCfgBlock start;
  private final PythonCfgBlock end = new PythonCfgEndBlock();
  private final Set<PythonCfgBlock> blocks = new HashSet<>();
  private final Deque<Loop> loops = new ArrayDeque<>();

  public ControlFlowGraphBuilder(@Nullable StatementList statementList) {
    blocks.add(end);
    if (statementList != null) {
      start = build(statementList.statements(), createSimpleBlock(end));
    } else {
      start = end;
    }
    removeEmptyBlocks();
  }

  private void removeEmptyBlocks() {
    Map<PythonCfgBlock, PythonCfgBlock> emptyBlockReplacements = new HashMap<>();
    for (PythonCfgBlock block : blocks) {
      if (block.isEmptyBlock()) {
        PythonCfgBlock firstNonEmptySuccessor = block.firstNonEmptySuccessor();
        emptyBlockReplacements.put(block, firstNonEmptySuccessor);
      }
    }

    blocks.removeAll(emptyBlockReplacements.keySet());

    for (PythonCfgBlock block : blocks) {
      block.replaceSuccessors(emptyBlockReplacements);
    }

    start = emptyBlockReplacements.getOrDefault(start, start);
  }

  public ControlFlowGraph getCfg() {
    return new ControlFlowGraph(Collections.unmodifiableSet(blocks), start, end);
  }

  private PythonCfgSimpleBlock createSimpleBlock(CfgBlock successor) {
    PythonCfgSimpleBlock block = new PythonCfgSimpleBlock(successor);
    blocks.add(block);
    return block;
  }

  private PythonCfgBranchingBlock createBranchingBlock(Tree branchingTree, CfgBlock trueSuccessor, CfgBlock falseSuccessor) {
    PythonCfgBranchingBlock block = new PythonCfgBranchingBlock(branchingTree, trueSuccessor, falseSuccessor);
    blocks.add(block);
    return block;
  }

  private PythonCfgBranchingBlock createBranchingBlock(Tree branchingTree, CfgBlock falseSuccessor) {
    PythonCfgBranchingBlock block = new PythonCfgBranchingBlock(branchingTree, null, falseSuccessor);
    blocks.add(block);
    return block;
  }

  private PythonCfgBlock build(List<Statement> statements, PythonCfgBlock successor) {
    PythonCfgBlock currentBlock = successor;
    for (int i = statements.size() - 1; i >= 0; i--) {
      Statement statement = statements.get(i);
      currentBlock = build(statement, currentBlock);
    }
    return currentBlock;
  }

  private PythonCfgBlock build(Statement statement, PythonCfgBlock currentBlock) {
    switch (statement.getKind()) {
      case RETURN_STMT:
        return buildReturnStatement((ReturnStatement) statement, currentBlock);
      case IF_STMT:
        return buildIfStatement(((IfStatement) statement), currentBlock);
      case WHILE_STMT:
        return buildWhileStatement(((WhileStatement) statement), currentBlock);
      case FOR_STMT:
        return buildForStatement(((ForStatement) statement), currentBlock);
      case CONTINUE_STMT:
        return buildContinueStatement(((ContinueStatement) statement), currentBlock);
      case TRY_STMT:
        return tryStatement(((TryStatement) statement), currentBlock);
      case BREAK_STMT:
        return buildBreakStatement((BreakStatement) statement, currentBlock);
      default:
        currentBlock.addElement(statement);
    }

    return currentBlock;
  }

  private PythonCfgBlock tryStatement(TryStatement tryStatement, PythonCfgBlock successor) {
    PythonCfgBlock finallyOrAfterTryBlock = successor;
    FinallyClause finallyClause = tryStatement.finallyClause();
    if (finallyClause != null) {
      finallyOrAfterTryBlock = build(finallyClause.body().statements(), createSimpleBlock(successor));
    }
    PythonCfgBlock firstExceptClauseBlock = exceptClauses(tryStatement, finallyOrAfterTryBlock);
    ElseStatement elseClause = tryStatement.elseClause();
    PythonCfgBlock tryBlockSuccessor = finallyOrAfterTryBlock;
    if (elseClause != null) {
      tryBlockSuccessor = build(elseClause.body().statements(), createSimpleBlock(finallyOrAfterTryBlock));
    }
    PythonCfgBlock firstTryBlock = build(tryStatement.body().statements(), createBranchingBlock(tryStatement, tryBlockSuccessor, firstExceptClauseBlock));
    return createSimpleBlock(firstTryBlock);
  }

  private PythonCfgBlock exceptClauses(TryStatement tryStatement, PythonCfgBlock finallyOrAfterTryBlock) {
    PythonCfgBlock falseSuccessor = finallyOrAfterTryBlock;
    List<ExceptClause> exceptClauses = tryStatement.exceptClauses();
    for (int i = exceptClauses.size() - 1; i >= 0; i--) {
      ExceptClause exceptClause = exceptClauses.get(i);
      PythonCfgBlock exceptBlock = build(exceptClause.body().statements(), createSimpleBlock(finallyOrAfterTryBlock));
      PythonCfgBlock exceptCondition = createBranchingBlock(exceptClause, exceptBlock, falseSuccessor);
      exceptCondition.addElement(exceptClause);
      falseSuccessor = exceptCondition;
    }
    return falseSuccessor;
  }

  private PythonCfgBlock buildBreakStatement(BreakStatement breakStatement, PythonCfgBlock syntacticSuccessor) {
    PythonCfgSimpleBlock block = createSimpleBlock(loops.peek().breakTarget);
    block.setSyntacticSuccessor(syntacticSuccessor);
    block.addElement(breakStatement);
    return block;
  }

  private PythonCfgBlock buildContinueStatement(ContinueStatement continueStatement, PythonCfgBlock syntacticSuccessor) {
    PythonCfgSimpleBlock block = createSimpleBlock(loops.peek().continueTarget);
    block.setSyntacticSuccessor(syntacticSuccessor);
    block.addElement(continueStatement);
    return block;
  }

  private PythonCfgBlock buildLoop(Tree branchingTree, Tree conditionElement, StatementList body, PythonCfgBlock successor) {
    PythonCfgBranchingBlock conditionBlock = createBranchingBlock(branchingTree, successor);
    conditionBlock.addElement(conditionElement);
    loops.push(new Loop(successor, conditionBlock));
    PythonCfgBlock whileBodyBlock = build(body.statements(), createSimpleBlock(conditionBlock));
    loops.pop();
    conditionBlock.setTrueSuccessor(whileBodyBlock);
    return createSimpleBlock(conditionBlock);
  }

  private PythonCfgBlock buildForStatement(ForStatement forStatement, PythonCfgBlock successor) {
    PythonCfgBlock beforeForStmt = buildLoop(forStatement, forStatement, forStatement.body(), successor);
    forStatement.testExpressions().forEach(beforeForStmt::addElement);
    return beforeForStmt;
  }

  private PythonCfgBlock buildWhileStatement(WhileStatement whileStatement, PythonCfgBlock currentBlock) {
    return buildLoop(whileStatement, whileStatement.condition(), whileStatement.body(), currentBlock);
  }

  /**
   * CFG for if-elif-else statement:
   *
   *                +-----------+
   *       +--------+ before_if +-------+
   *       |        +-----------+       |
   *       |                            |
   * +-----v----+                +------v-----+
   * | if_body  |          +-----+ elif_cond  +-----+
   * +----+-----+          |     +------------+     |
   *      |                |                        |
   *      |          +-----v-----+            +-----v-----+
   *      |          | elif_body |            | else_body |
   *      |          +-----+-----+            +-----+-----+
   *      |                |                        |
   *      |        +-------v-----+                  |
   *      +-------->  after_if   <------------------+
   *               +-------------+
   */
  private PythonCfgBlock buildIfStatement(IfStatement ifStatement, PythonCfgBlock afterBlock) {
    PythonCfgBlock ifBodyBlock = createSimpleBlock(afterBlock);
    ifBodyBlock = build(ifStatement.body().statements(), ifBodyBlock);
    ElseStatement elseClause = ifStatement.elseBranch();
    PythonCfgBlock falseSuccessor = afterBlock;
    if (elseClause != null) {
      PythonCfgBlock elseBodyBlock = createSimpleBlock(afterBlock);
      elseBodyBlock = build(elseClause.body().statements(), elseBodyBlock);
      falseSuccessor = elseBodyBlock;
    }
    falseSuccessor = buildElifClauses(afterBlock, falseSuccessor, ifStatement.elifBranches());
    PythonCfgBlock beforeIfBlock = createBranchingBlock(ifStatement, ifBodyBlock, falseSuccessor);
    beforeIfBlock.addElement(ifStatement.condition());
    return beforeIfBlock;
  }

  private PythonCfgBlock buildElifClauses(PythonCfgBlock currentBlock, PythonCfgBlock falseSuccessor, List<IfStatement> elifBranches) {
    for (int i = elifBranches.size() - 1; i >= 0; i--) {
      IfStatement elifStatement = elifBranches.get(i);
      PythonCfgBlock elifBodyBlock = createSimpleBlock(currentBlock);
      elifBodyBlock = build(elifStatement.body().statements(), elifBodyBlock);
      PythonCfgBlock beforeElifBlock = createBranchingBlock(elifStatement, elifBodyBlock, falseSuccessor);
      beforeElifBlock.addElement(elifStatement.condition());
      falseSuccessor = beforeElifBlock;
    }
    return falseSuccessor;
  }

  private PythonCfgBlock buildReturnStatement(ReturnStatement statement, PythonCfgBlock syntacticSuccessor) {
    PythonCfgSimpleBlock block = createSimpleBlock(end);
    block.setSyntacticSuccessor(syntacticSuccessor);
    block.addElement(statement);
    return block;
  }

  private static class Loop {

    final PythonCfgBlock breakTarget;
    final PythonCfgBlock continueTarget;

    private Loop(PythonCfgBlock breakTarget, PythonCfgBlock continueTarget) {
      this.breakTarget = breakTarget;
      this.continueTarget = continueTarget;
    }
  }

}
