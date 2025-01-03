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
package org.sonar.python.cfg;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.BreakStatement;
import org.sonar.plugins.python.api.tree.CaseBlock;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ContinueStatement;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FinallyClause;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Guard;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.MatchStatement;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.TupleParameter;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.python.tree.TreeUtils;

public class ControlFlowGraphBuilder {

  private PythonCfgBlock start;
  private final PythonCfgBlock end = new PythonCfgEndBlock();
  private final Set<PythonCfgBlock> blocks = new LinkedHashSet<>();
  private final Deque<Loop> loops = new ArrayDeque<>();
  private final Deque<PythonCfgBlock> exceptionTargets = new ArrayDeque<>();
  private final Deque<PythonCfgBlock> exitTargets = new ArrayDeque<>();

  public ControlFlowGraphBuilder(@Nullable StatementList statementList) {
    blocks.add(end);
    exceptionTargets.push(end);
    exitTargets.push(end);
    if (statementList != null) {
      start = build(statementList.statements(), createSimpleBlock(end));
      addParametersToStartBlock(statementList);
    } else {
      start = end;
    }
    removeEmptyBlocks();
    computePredecessors();
  }

  private void addParametersToStartBlock(StatementList statementList) {
    if (statementList.parent().is(Tree.Kind.FUNCDEF)) {
      ParameterList parameterList = ((FunctionDef) statementList.parent()).parameters();
      if (parameterList != null) {
        PythonCfgSimpleBlock parametersBlock = createSimpleBlock(start);
        addParameters(parameterList.all(), parametersBlock);
        start = parametersBlock;
      }
    }
  }

  private static void addParameters(List<AnyParameter> parameters, PythonCfgSimpleBlock parametersBlock) {
    for (AnyParameter parameter : parameters) {
      if (parameter.is(Tree.Kind.TUPLE_PARAMETER)) {
        addParameters(((TupleParameter) parameter).parameters(), parametersBlock);
      } else {
        parametersBlock.addElement(parameter);
      }
    }
  }

  private void computePredecessors() {
    for (PythonCfgBlock block : blocks) {
      for (CfgBlock successor : block.successors()) {
        ((PythonCfgBlock) successor).addPredecessor(block);
      }
    }
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
      case WITH_STMT:
        return buildWithStatement((WithStatement) statement, currentBlock);
      case CLASSDEF:
        return buildClassDefStatement((ClassDef) statement, currentBlock);
      case RETURN_STMT:
        return buildReturnStatement((ReturnStatement) statement, currentBlock);
      case RAISE_STMT:
        return buildRaiseStatement((RaiseStatement) statement, currentBlock);
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
      case MATCH_STMT:
        return buildMatchStatement((MatchStatement) statement, currentBlock);
      case FUNCDEF:
        return buildFuncDefStatement((FunctionDef) statement, currentBlock);
      default:
        currentBlock.addElement(statement);
    }

    return currentBlock;
  }

  private PythonCfgBlock buildClassDefStatement(ClassDef classDef, PythonCfgBlock currentBlock) {
    PythonCfgBlock block = build(classDef.body().statements(), currentBlock);
    block.addElement(classDef);
    classDef.decorators().stream().forEach(currentBlock::addElement);
    return block;
  }

  private static PythonCfgBlock buildFuncDefStatement(FunctionDef functionDef, PythonCfgBlock currentBlock) {
    currentBlock.addElement(functionDef);
    functionDef.decorators().stream().forEach(currentBlock::addElement);
    return currentBlock;
  }

  private PythonCfgBlock buildMatchStatement(MatchStatement statement, PythonCfgBlock successor) {
    List<CaseBlock> caseBlocks = statement.caseBlocks();
    PythonCfgBlock matchingBlock = null;
    PythonCfgBlock falseSuccessor = successor;
    for (int i = caseBlocks.size() - 1; i >= 0; i--) {
      PythonCfgBlock caseBodyBlock = createSimpleBlock(successor);
      CaseBlock caseBlock = caseBlocks.get(i);
      Pattern pattern = caseBlock.pattern();
      Guard guard = caseBlock.guard();
      caseBodyBlock = build(caseBlock.body().statements(), caseBodyBlock);
      matchingBlock = createBranchingBlock(pattern, caseBodyBlock, falseSuccessor);
      if (guard != null) {
        matchingBlock.addElement(guard.condition());
      }
      matchingBlock.addElement(pattern);
      matchingBlock.addElement(statement.subjectExpression());
      blocks.add(matchingBlock);
      falseSuccessor = matchingBlock;
    }
    return matchingBlock;
  }

  private PythonCfgBlock buildWithStatement(WithStatement withStatement, PythonCfgBlock successor) {
    PythonCfgBlock withBodyBlock = build(withStatement.statements().statements(), createSimpleBlock(successor));
    // exceptions may be raised inside with block and be caught by context manager
    // see https://docs.python.org/3/reference/compound_stmts.html#the-with-statement
    PythonCfgBranchingBlock branchingBlock = createBranchingBlock(withStatement, withBodyBlock, successor);
    for (int i = withStatement.withItems().size() - 1; i >= 0; i--) {
      branchingBlock.addElement(withStatement.withItems().get(i));
    }
    return branchingBlock;
  }

  private PythonCfgBlock tryStatement(TryStatement tryStatement, PythonCfgBlock successor) {
    PythonCfgBlock finallyOrAfterTryBlock = successor;
    FinallyClause finallyClause = tryStatement.finallyClause();
    PythonCfgBlock finallyBlock = null;
    if (finallyClause != null) {
      finallyOrAfterTryBlock = build(finallyClause.body().statements(), createBranchingBlock(finallyClause, successor, exitTargets.peek()));
      finallyBlock = finallyOrAfterTryBlock;
      exitTargets.push(finallyBlock);
      loops.push(new Loop(finallyBlock, finallyBlock));
    }
    PythonCfgBlock firstExceptClauseBlock = exceptClauses(tryStatement, finallyOrAfterTryBlock, finallyBlock);
    ElseClause elseClause = tryStatement.elseClause();
    PythonCfgBlock tryBlockSuccessor = finallyOrAfterTryBlock;
    if (elseClause != null) {
      tryBlockSuccessor = build(elseClause.body().statements(), createSimpleBlock(finallyOrAfterTryBlock));
    }
    if (finallyClause != null) {
      exitTargets.pop();
      loops.pop();
    }
    exceptionTargets.push(firstExceptClauseBlock);
    exitTargets.push(firstExceptClauseBlock);
    loops.push(new Loop(firstExceptClauseBlock, firstExceptClauseBlock));
    PythonCfgBlock firstTryBlock = build(tryStatement.body().statements(), createBranchingBlock(tryStatement, tryBlockSuccessor, firstExceptClauseBlock));
    exceptionTargets.pop();
    exitTargets.pop();
    loops.pop();
    return createSimpleBlock(firstTryBlock);
  }

  private PythonCfgBlock exceptClauses(TryStatement tryStatement, PythonCfgBlock finallyOrAfterTryBlock, @Nullable PythonCfgBlock finallyBlock) {
    PythonCfgBlock falseSuccessor = finallyBlock == null ? exceptionTargets.peek() : finallyBlock;
    List<ExceptClause> exceptClauses = tryStatement.exceptClauses();
    for (int i = exceptClauses.size() - 1; i >= 0; i--) {
      ExceptClause exceptClause = exceptClauses.get(i);
      PythonCfgBlock exceptBlock = build(exceptClause.body().statements(), createSimpleBlock(finallyOrAfterTryBlock));
      PythonCfgBlock exceptCondition = createBranchingBlock(exceptClause, exceptBlock, falseSuccessor);
      Expression exceptionInstance = exceptClause.exceptionInstance();
      if (exceptionInstance != null) {
        exceptCondition.addElement(exceptionInstance);
      }
      Expression exception = exceptClause.exception();
      if (exception != null) {
        exceptCondition.addElement(exception);
      }
      falseSuccessor = exceptCondition;
    }
    return falseSuccessor;
  }

  private Loop currentLoop(Tree tree) {
    Loop loop = loops.peek();
    if (loop == null) {
      Token token = tree.firstToken();
      throw new IllegalStateException("Invalid \"" + token.value() + "\" outside loop at line " + token.line());
    }
    return loop;
  }

  private PythonCfgBlock buildBreakStatement(BreakStatement breakStatement, PythonCfgBlock syntacticSuccessor) {
    PythonCfgSimpleBlock block = createSimpleBlock(currentLoop(breakStatement).breakTarget);
    block.setSyntacticSuccessor(syntacticSuccessor);
    block.addElement(breakStatement);
    return block;
  }

  private PythonCfgBlock buildContinueStatement(ContinueStatement continueStatement, PythonCfgBlock syntacticSuccessor) {
    PythonCfgSimpleBlock block = createSimpleBlock(currentLoop(continueStatement).continueTarget);
    block.setSyntacticSuccessor(syntacticSuccessor);
    block.addElement(continueStatement);
    return block;
  }

  private PythonCfgBlock buildLoop(Tree branchingTree, List<Expression> conditionElements, StatementList body, @Nullable ElseClause elseClause, PythonCfgBlock successor) {
    PythonCfgBlock afterLoopBlock = successor;
    if (elseClause != null) {
      afterLoopBlock = build(elseClause.body().statements(), createSimpleBlock(successor));
    }
    PythonCfgBranchingBlock conditionBlock = createBranchingBlock(branchingTree, afterLoopBlock);
    conditionElements.forEach(conditionBlock::addElement);
    loops.push(new Loop(successor, conditionBlock));
    PythonCfgBlock loopBodyBlock = build(body.statements(), createSimpleBlock(conditionBlock));
    loops.pop();
    conditionBlock.setTrueSuccessor(loopBodyBlock);
    return createSimpleBlock(conditionBlock);
  }

  private PythonCfgBlock buildForStatement(ForStatement forStatement, PythonCfgBlock successor) {
    PythonCfgBlock beforeForStmt = buildLoop(forStatement, forStatement.expressions(), forStatement.body(), forStatement.elseClause(), successor);
    forStatement.testExpressions().forEach(beforeForStmt::addElement);
    return beforeForStmt;
  }

  private PythonCfgBlock buildWhileStatement(WhileStatement whileStatement, PythonCfgBlock currentBlock) {
    return buildLoop(whileStatement, Collections.singletonList(whileStatement.condition()), whileStatement.body(), whileStatement.elseClause(), currentBlock);
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
    ElseClause elseClause = ifStatement.elseBranch();
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
    if (TreeUtils.firstAncestorOfKind(statement, Tree.Kind.FUNCDEF) == null || isStatementAtClassLevel(statement)) {
      throw new IllegalStateException("Invalid return outside of a function");
    }
    PythonCfgSimpleBlock block = createSimpleBlock(exitTargets.peek());
    block.setSyntacticSuccessor(syntacticSuccessor);
    block.addElement(statement);
    return block;
  }

  // assumption: parent of return statement is always a statementList, which, in turn, has always a parent
  private static boolean isStatementAtClassLevel(ReturnStatement statement) {
    return statement.parent().parent().is(Tree.Kind.CLASSDEF);
  }

  private PythonCfgBlock buildRaiseStatement(RaiseStatement statement, PythonCfgBlock syntacticSuccessor) {
    PythonCfgSimpleBlock block = createSimpleBlock(exceptionTargets.peek());
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
