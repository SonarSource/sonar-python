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

import com.intellij.psi.PsiElement;
import com.intellij.psi.tree.IElementType;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.PyTokenTypes;
import com.jetbrains.python.psi.PyBinaryExpression;
import com.jetbrains.python.psi.PyElementType;
import com.jetbrains.python.psi.PyExpression;
import com.jetbrains.python.psi.PyFunction;
import com.jetbrains.python.psi.PyRecursiveElementVisitor;
import com.jetbrains.python.psi.PyReturnStatement;
import com.jetbrains.python.psi.PyStatementListContainer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

public class CognitiveComplexityVisitor extends PyRecursiveElementVisitor {

  private int complexity = 0;
  private Deque<NestingLevel> nestingLevelStack = new LinkedList<>();
  private Set<PsiElement> alreadyConsideredOperators = new HashSet<>();

  @Nullable
  private final SecondaryLocationConsumer secondaryLocationConsumer;

  public interface SecondaryLocationConsumer {
    void consume(PsiElement element, String message);
  }

  CognitiveComplexityVisitor(@Nullable SecondaryLocationConsumer secondaryLocationConsumer) {
    this.secondaryLocationConsumer = secondaryLocationConsumer;
    nestingLevelStack.push(new NestingLevel());
  }

  public static int complexity(PsiElement element, @Nullable SecondaryLocationConsumer secondaryLocationConsumer) {
    CognitiveComplexityVisitor visitor = new CognitiveComplexityVisitor(secondaryLocationConsumer);
    element.accept(visitor);
    return visitor.complexity;
  }

  public int getComplexity() {
    return complexity;
  }

  private static final Set<IElementType> TYPES_INCREMENTING_WITH_NESTING = new HashSet<>(Arrays.asList(
    PyElementTypes.IF_STATEMENT,
    PyElementTypes.WHILE_STATEMENT,
    PyElementTypes.FOR_STATEMENT,
    PyElementTypes.EXCEPT_PART
  ));

  private static final Set<IElementType> TYPES_INCREMENTING_WITHOUT_NESTING = new HashSet<>(Arrays.asList(
    PyElementTypes.IF_PART_ELIF,
    PyElementTypes.ELSE_PART
  ));

  private static final Set<IElementType> NON_NESTING_STATEMENT_LISTS = new HashSet<>(Arrays.asList(
    PyElementTypes.TRY_PART,
    PyElementTypes.FINALLY_PART,
    PyElementTypes.CLASS_DECLARATION,
    PyElementTypes.FUNCTION_DECLARATION,
    PyElementTypes.WITH_STATEMENT
  ));

  @Override
  public void visitElement(PsiElement element) {
    IElementType elementType = element.getNode().getElementType();

    if (TYPES_INCREMENTING_WITH_NESTING.contains(elementType)) {
      incrementWithNesting(element.getNode().findLeafElementAt(0).getPsi());
    } else if (TYPES_INCREMENTING_WITHOUT_NESTING.contains(elementType)) {
      incrementWithoutNesting(element.getNode().findLeafElementAt(0).getPsi());
    } else if (isLogicalBinaryExpression(element)) {
      visitLogicalBinaryExpression((PyBinaryExpression) element);
    }

    if (elementType == PyElementTypes.FUNCTION_DECLARATION || elementType == PyElementTypes.CLASS_DECLARATION) {
      nestingLevelStack.push(new NestingLevel(nestingLevelStack.peek(), element));
    } else if (isStatementListIncrementingNestingLevel(element)) {
      nestingLevelStack.peek().increment();
    } else if (elementType == PyElementTypes.CONDITIONAL_EXPRESSION) {
      incrementWithNesting(element.getNode().findChildByType(PyTokenTypes.IF_KEYWORD).getPsi());
      nestingLevelStack.peek().increment();
    }

    super.visitElement(element);

    if (elementType == PyElementTypes.FUNCTION_DECLARATION || elementType == PyElementTypes.CLASS_DECLARATION) {
      nestingLevelStack.pop();
    } else if (isStatementListIncrementingNestingLevel(element)) {
      nestingLevelStack.peek().decrement();
    } else if (elementType == PyElementTypes.CONDITIONAL_EXPRESSION) {
      nestingLevelStack.peek().decrement();
    }
  }

  private static boolean isLogicalBinaryExpression(PsiElement element) {
    if (element instanceof PyBinaryExpression) {
      PyBinaryExpression binaryExpression = (PyBinaryExpression) element;
      PyElementType operator = binaryExpression.getOperator();
      return operator == PyTokenTypes.AND_KEYWORD || operator == PyTokenTypes.OR_KEYWORD;
    }
    return false;
  }

  private void visitLogicalBinaryExpression(PyBinaryExpression tree) {
    if (alreadyConsideredOperators.contains(tree.getPsiOperator())) {
      return;
    }

    List<PsiElement> operators = new ArrayList<>();
    flattenOperators(tree, operators);

    PsiElement previous = null;
    for (PsiElement operator : operators) {
      if (previous == null || !previous.getNode().getElementType().equals(operator.getNode().getElementType())) {
        incrementWithoutNesting(operator);
      }
      previous = operator;
      alreadyConsideredOperators.add(operator);
    }
  }

  private static void flattenOperators(PyBinaryExpression binaryExpression, List<PsiElement> operators) {
    PyExpression left = binaryExpression.getLeftExpression();
    if (isLogicalBinaryExpression(left)) {
      flattenOperators((PyBinaryExpression) left, operators);
    }

    operators.add(binaryExpression.getPsiOperator());

    PyExpression right = binaryExpression.getRightExpression();
    if (isLogicalBinaryExpression(right)) {
      flattenOperators((PyBinaryExpression) right, operators);
    }
  }

  private static boolean isStatementListIncrementingNestingLevel(PsiElement element) {
    if (element instanceof PyStatementListContainer) {
      IElementType elementType = element.getNode().getElementType();
      return !NON_NESTING_STATEMENT_LISTS.contains(elementType);
    }
    return false;
  }

  private void incrementWithNesting(PsiElement secondaryLocationNode) {
    incrementComplexity(secondaryLocationNode, 1 + nestingLevelStack.peek().level());
  }

  private void incrementWithoutNesting(PsiElement secondaryLocationNode) {
    incrementComplexity(secondaryLocationNode, 1);
  }

  private void incrementComplexity(PsiElement secondaryLocationNode, int currentNodeComplexity) {
    if (secondaryLocationConsumer != null) {
      secondaryLocationConsumer.consume(secondaryLocationNode, secondaryMessage(currentNodeComplexity));
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
    private PsiElement element;
    private int level;

    private NestingLevel() {
      element = null;
      level = 0;
    }

    private NestingLevel(NestingLevel parent, PsiElement element) {
      this.element = element;
      if (isFunction()) {
        if (parent.isWrapperFunction(element)) {
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
      return element != null && element.getNode().getElementType() == PyElementTypes.FUNCTION_DECLARATION;
    }

    private boolean isWrapperFunction(PsiElement childFunction) {
      if (isFunction()) {
        return Arrays.stream(((PyFunction) element).getStatementList().getStatements())
          .filter(statement -> statement != childFunction)
          .allMatch(NestingLevel::isSimpleReturn);
      }
      return false;
    }

    private static boolean isSimpleReturn(PsiElement statement) {
      if (statement instanceof PyReturnStatement) {
        PyReturnStatement returnStatement = (PyReturnStatement) statement;
        return returnStatement.getExpression().getNode().getElementType() == PyElementTypes.REFERENCE_EXPRESSION;
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
