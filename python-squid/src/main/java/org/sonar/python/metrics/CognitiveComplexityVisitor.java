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
package org.sonar.python.metrics;

import com.sonar.sslr.api.Token;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyConditionalExpressionTree;
import org.sonar.python.api.tree.PyElseStatementTree;
import org.sonar.python.api.tree.PyExceptClauseTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyReturnStatementTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.PyWhileStatementTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.tree.BaseTreeVisitor;

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
  public void visitIfStatement(PyIfStatementTree pyIfStatementTree) {
    if (pyIfStatementTree.isElif()) {
      incrementWithoutNesting(pyIfStatementTree.keyword());
    } else {
      incrementWithNesting(pyIfStatementTree.keyword());
    }
    super.visitIfStatement(pyIfStatementTree);
  }

  @Override
  public void visitElseStatement(PyElseStatementTree pyElseStatementTree) {
    incrementWithoutNesting(pyElseStatementTree.elseKeyword());
    super.visitElseStatement(pyElseStatementTree);
  }

  @Override
  public void visitWhileStatement(PyWhileStatementTree pyWhileStatementTree) {
    incrementWithNesting(pyWhileStatementTree.whileKeyword());
    if (pyWhileStatementTree.elseKeyword() != null) {
      incrementWithoutNesting(pyWhileStatementTree.elseKeyword());
    }
    super.visitWhileStatement(pyWhileStatementTree);
  }

  @Override
  public void visitForStatement(PyForStatementTree pyForStatementTree) {
    incrementWithNesting(pyForStatementTree.forKeyword());
    if (pyForStatementTree.elseKeyword() != null) {
      incrementWithoutNesting(pyForStatementTree.elseKeyword());
    }
    super.visitForStatement(pyForStatementTree);
  }

  @Override
  public void visitExceptClause(PyExceptClauseTree pyExceptClauseTree) {
    incrementWithNesting(pyExceptClauseTree.exceptKeyword());
    super.visitExceptClause(pyExceptClauseTree);
  }

  @Override
  public void visitBinaryExpression(PyBinaryExpressionTree pyBinaryExpressionTree) {
    if (pyBinaryExpressionTree.is(Kind.AND) || pyBinaryExpressionTree.is(Kind.OR)) {
      if (alreadyConsideredOperators.contains(pyBinaryExpressionTree.operator())) {
        super.visitBinaryExpression(pyBinaryExpressionTree);
        return;
      }
      List<Token> operators = new ArrayList<>();
      flattenOperators(pyBinaryExpressionTree, operators);
      Token previous = null;
      for (Token operator : operators) {
        if (previous == null || !previous.getType().equals(operator.getType())) {
          incrementWithoutNesting(pyBinaryExpressionTree.operator());
        }
        previous = operator;
        alreadyConsideredOperators.add(operator);
      }
    }
    super.visitBinaryExpression(pyBinaryExpressionTree);
  }

  private void flattenOperators(PyBinaryExpressionTree binaryExpression, List<Token> operators) {
    PyExpressionTree left = binaryExpression.leftOperand();
    if (left.is(Kind.AND) || left.is(Kind.OR)) {
      flattenOperators((PyBinaryExpressionTree) left, operators);
    }

    operators.add(binaryExpression.operator());

    PyExpressionTree right = binaryExpression.rightOperand();
    if (right.is(Kind.AND) || right.is(Kind.OR)) {
      flattenOperators((PyBinaryExpressionTree) right, operators);
    }
  }

  @Override
  public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
    nestingLevelStack.push(new NestingLevel(nestingLevelStack.peek(), pyFunctionDefTree));
    super.visitFunctionDef(pyFunctionDefTree);
    nestingLevelStack.pop();
  }

  @Override
  public void visitClassDef(PyClassDefTree pyClassDefTree) {
    nestingLevelStack.push(new NestingLevel(nestingLevelStack.peek(), pyClassDefTree));
    super.visitClassDef(pyClassDefTree);
    nestingLevelStack.pop();
  }

  @Override
  public void visitStatementList(PyStatementListTree pyStatementListTree) {
    if (isStmtListIncrementsNestingLevel(pyStatementListTree)) {
      nestingLevelStack.peek().increment();
      super.visitStatementList(pyStatementListTree);
      nestingLevelStack.peek().decrement();
    } else {
      super.visitStatementList(pyStatementListTree);
    }
  }

  @Override
  public void visitConditionalExpression(PyConditionalExpressionTree pyConditionalExpressionTree) {
    incrementWithNesting(pyConditionalExpressionTree.ifKeyword());
    nestingLevelStack.peek().increment();
    super.visitConditionalExpression(pyConditionalExpressionTree);
    nestingLevelStack.peek().decrement();
  }

  private static boolean isStmtListIncrementsNestingLevel(PyStatementListTree statementListTree) {
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
        if (parent.isWrapperFunction((PyFunctionDefTree) tree)) {
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

    private boolean isWrapperFunction(PyFunctionDefTree childFunction) {
      if(tree != null && tree.is(Kind.FUNCDEF)) {
        return ((PyFunctionDefTree) tree).body()
          .statements()
          .stream()
          .filter(statement -> statement != childFunction)
          .allMatch(NestingLevel::isSimpleReturn);
      }
      return false;
    }

    private static boolean isSimpleReturn(PyStatementTree statement) {
      if (statement.is(Kind.RETURN_STMT)) {
        PyReturnStatementTree returnStatementTree = (PyReturnStatementTree) statement;
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
