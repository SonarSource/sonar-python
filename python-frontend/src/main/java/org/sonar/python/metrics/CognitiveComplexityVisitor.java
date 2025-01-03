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
package org.sonar.python.metrics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;

public class CognitiveComplexityVisitor extends BaseTreeVisitor {

  private int complexity = 0;
  private Deque<NestingLevel> nestingLevelStack = new LinkedList<>();
  private Set<Token> alreadyConsideredOperators = new HashSet<>();

  @Nullable
  private final SecondaryLocationConsumer secondaryLocationConsumer;

  public interface SecondaryLocationConsumer {
    void consume(Token secondaryLocation, String message);
  }

  CognitiveComplexityVisitor(@Nullable SecondaryLocationConsumer secondaryLocationConsumer) {
    this.secondaryLocationConsumer = secondaryLocationConsumer;
    nestingLevelStack.push(new NestingLevel());
  }

  public static int complexity(Tree tree, @Nullable SecondaryLocationConsumer secondaryLocationConsumer) {
    CognitiveComplexityVisitor visitor = new CognitiveComplexityVisitor(secondaryLocationConsumer);
    visitor.scan(tree);
    return visitor.complexity;
  }

  public int getComplexity() {
    return complexity;
  }

  @Override
  public void visitIfStatement(IfStatement pyIfStatementTree) {
    if (pyIfStatementTree.isElif()) {
      incrementWithoutNesting(pyIfStatementTree.keyword());
    } else {
      incrementWithNesting(pyIfStatementTree.keyword());
    }
    super.visitIfStatement(pyIfStatementTree);
  }

  @Override
  public void visitElseClause(ElseClause pyElseClauseTree) {
    incrementWithoutNesting(pyElseClauseTree.elseKeyword());
    super.visitElseClause(pyElseClauseTree);
  }

  @Override
  public void visitWhileStatement(WhileStatement pyWhileStatementTree) {
    incrementWithNesting(pyWhileStatementTree.whileKeyword());
    super.visitWhileStatement(pyWhileStatementTree);
  }

  @Override
  public void visitForStatement(ForStatement pyForStatementTree) {
    incrementWithNesting(pyForStatementTree.forKeyword());
    super.visitForStatement(pyForStatementTree);
  }

  @Override
  public void visitExceptClause(ExceptClause exceptClause) {
    incrementWithNesting(exceptClause.exceptKeyword());
    super.visitExceptClause(exceptClause);
  }

  @Override
  public void visitBinaryExpression(BinaryExpression pyBinaryExpressionTree) {
    if (pyBinaryExpressionTree.is(Kind.AND) || pyBinaryExpressionTree.is(Kind.OR)) {
      if (alreadyConsideredOperators.contains(pyBinaryExpressionTree.operator())) {
        super.visitBinaryExpression(pyBinaryExpressionTree);
        return;
      }
      List<Token> operators = new ArrayList<>();
      flattenOperators(pyBinaryExpressionTree, operators);
      Token previous = null;
      for (Token operator : operators) {
        if (previous == null || !previous.type().equals(operator.type())) {
          incrementWithoutNesting(pyBinaryExpressionTree.operator());
        }
        previous = operator;
        alreadyConsideredOperators.add(operator);
      }
    }
    super.visitBinaryExpression(pyBinaryExpressionTree);
  }

  private static void flattenOperators(BinaryExpression binaryExpression, List<Token> operators) {
    Expression left = binaryExpression.leftOperand();
    if (left.is(Kind.AND) || left.is(Kind.OR)) {
      flattenOperators((BinaryExpression) left, operators);
    }

    operators.add(binaryExpression.operator());

    Expression right = binaryExpression.rightOperand();
    if (right.is(Kind.AND) || right.is(Kind.OR)) {
      flattenOperators((BinaryExpression) right, operators);
    }
  }

  @Override
  public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
    nestingLevelStack.push(new NestingLevel(nestingLevelStack.peek(), pyFunctionDefTree));
    super.visitFunctionDef(pyFunctionDefTree);
    nestingLevelStack.pop();
  }

  @Override
  public void visitClassDef(ClassDef pyClassDefTree) {
    nestingLevelStack.push(new NestingLevel(nestingLevelStack.peek(), pyClassDefTree));
    super.visitClassDef(pyClassDefTree);
    nestingLevelStack.pop();
  }

  @Override
  public void visitStatementList(StatementList statementList) {
    if (isStmtListIncrementsNestingLevel(statementList)) {
      nestingLevelStack.peek().increment();
      super.visitStatementList(statementList);
      nestingLevelStack.peek().decrement();
    } else {
      super.visitStatementList(statementList);
    }
  }

  @Override
  public void visitConditionalExpression(ConditionalExpression pyConditionalExpressionTree) {
    incrementWithNesting(pyConditionalExpressionTree.ifKeyword());
    nestingLevelStack.peek().increment();
    super.visitConditionalExpression(pyConditionalExpressionTree);
    nestingLevelStack.peek().decrement();
  }

  private static boolean isStmtListIncrementsNestingLevel(StatementList statementListTree) {
    if (statementListTree.parent().is(Kind.FILE_INPUT)) {
      return false;
    }
    List<Kind> notIncrementingNestingKinds = Arrays.asList(Kind.TRY_STMT, Kind.FINALLY_CLAUSE, Kind.CLASSDEF, Kind.FUNCDEF, Kind.WITH_STMT);
    return statementListTree.parent() != null && notIncrementingNestingKinds.stream().noneMatch(kind -> statementListTree.parent().is(kind));
  }

  private void incrementWithNesting(Token secondaryLocation) {
    incrementComplexity(secondaryLocation, 1 + nestingLevelStack.peek().level());
  }

  private void incrementWithoutNesting(Token secondaryLocation) {
    incrementComplexity(secondaryLocation, 1);
  }

  private void incrementComplexity(Token secondaryLocation, int currentNodeComplexity) {
    if (secondaryLocationConsumer != null) {
      secondaryLocationConsumer.consume(secondaryLocation, secondaryMessage(currentNodeComplexity));
    }
    complexity += currentNodeComplexity;
  }

  private static String secondaryMessage(int complexity) {
    if (complexity == 1) {
      return "+1";
    } else {
      return String.format("+%s (incl %s for nesting)", complexity, complexity - 1);
    }
  }

  private static class NestingLevel {

    @Nullable
    private Tree tree;
    private int level;

    private NestingLevel() {
      tree = null;
      level = 0;
    }

    private NestingLevel(NestingLevel parent, Tree tree) {
      this.tree = tree;
      if (tree.is(Kind.FUNCDEF)) {
        if (parent.isWrapperFunction((FunctionDef) tree)) {
          level = parent.level;
        } else if (parent.isFunction()) {
          level = parent.level + 1;
        } else {
          level = 0;
        }
      } else {
        // PythonGrammar.CLASSDEF
        level = 0;
      }
    }

    private boolean isFunction() {
      return tree != null && tree.is(Kind.FUNCDEF);
    }

    private boolean isWrapperFunction(FunctionDef childFunction) {
      if(tree != null && tree.is(Kind.FUNCDEF)) {
        return ((FunctionDef) tree).body()
          .statements()
          .stream()
          .filter(statement -> statement != childFunction)
          .allMatch(NestingLevel::isSimpleReturn);
      }
      return false;
    }

    private static boolean isSimpleReturn(Statement statement) {
      if (statement.is(Kind.RETURN_STMT)) {
        ReturnStatement returnStatementTree = (ReturnStatement) statement;
        return returnStatementTree.expressions().size() == 1 && returnStatementTree.expressions().get(0).is(Kind.NAME);
      }
      return false;
    }

    private int level() {
      return level;
    }

    private void increment() {
      level++;
    }

    private void decrement() {
      level--;
    }

  }

}
