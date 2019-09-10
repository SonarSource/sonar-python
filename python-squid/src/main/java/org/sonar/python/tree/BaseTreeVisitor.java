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
package org.sonar.python.tree;

import java.util.List;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyArgListTree;
import org.sonar.python.api.tree.PyAnnotatedAssignmentTree;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyAssertStatementTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyAwaitExpressionTree;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyBreakStatementTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyCompoundAssignmentStatementTree;
import org.sonar.python.api.tree.PyComprehensionForTree;
import org.sonar.python.api.tree.PyComprehensionIfTree;
import org.sonar.python.api.tree.PyConditionalExpressionTree;
import org.sonar.python.api.tree.PyContinueStatementTree;
import org.sonar.python.api.tree.PyDecoratorTree;
import org.sonar.python.api.tree.PyDelStatementTree;
import org.sonar.python.api.tree.PyDictionaryLiteralTree;
import org.sonar.python.api.tree.PyDottedNameTree;
import org.sonar.python.api.tree.PyEllipsisExpressionTree;
import org.sonar.python.api.tree.PyElseStatementTree;
import org.sonar.python.api.tree.PyExceptClauseTree;
import org.sonar.python.api.tree.PyExecStatementTree;
import org.sonar.python.api.tree.PyExpressionListTree;
import org.sonar.python.api.tree.PyExpressionStatementTree;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.PyFinallyClauseTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyGlobalStatementTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyImportFromTree;
import org.sonar.python.api.tree.PyImportNameTree;
import org.sonar.python.api.tree.PyKeyValuePairTree;
import org.sonar.python.api.tree.PyLambdaExpressionTree;
import org.sonar.python.api.tree.PyListLiteralTree;
import org.sonar.python.api.tree.PyComprehensionExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyNoneExpressionTree;
import org.sonar.python.api.tree.PyNonlocalStatementTree;
import org.sonar.python.api.tree.PyNumericLiteralTree;
import org.sonar.python.api.tree.PyParameterTree;
import org.sonar.python.api.tree.PyParenthesizedExpressionTree;
import org.sonar.python.api.tree.PyPassStatementTree;
import org.sonar.python.api.tree.PyPrintStatementTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.PyRaiseStatementTree;
import org.sonar.python.api.tree.PyReprExpressionTree;
import org.sonar.python.api.tree.PyReturnStatementTree;
import org.sonar.python.api.tree.PySetLiteralTree;
import org.sonar.python.api.tree.PySliceExpressionTree;
import org.sonar.python.api.tree.PySliceItemTree;
import org.sonar.python.api.tree.PySliceListTree;
import org.sonar.python.api.tree.PyStarredExpressionTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyStringElementTree;
import org.sonar.python.api.tree.PyStringLiteralTree;
import org.sonar.python.api.tree.PySubscriptionExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyTryStatementTree;
import org.sonar.python.api.tree.PyTupleParameterTree;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.PyTypeAnnotationTree;
import org.sonar.python.api.tree.PyParameterListTree;
import org.sonar.python.api.tree.PyUnaryExpressionTree;
import org.sonar.python.api.tree.PyWhileStatementTree;
import org.sonar.python.api.tree.PyWithItemTree;
import org.sonar.python.api.tree.PyWithStatementTree;
import org.sonar.python.api.tree.PyYieldExpressionTree;
import org.sonar.python.api.tree.PyYieldStatementTree;
import org.sonar.python.api.tree.Tree;

/**
 * Default implementation of {@link org.sonar.python.api.tree.PyTreeVisitor}.
 */
public class BaseTreeVisitor implements PyTreeVisitor {

  protected void scan(@Nullable Tree tree) {
    if (tree != null) {
      tree.accept(this);
    }
  }

  protected void scan(List<? extends Tree> trees) {
    if (trees != null) {
      for (Tree tree : trees) {
        scan(tree);
      }
    }
  }

  @Override
  public void visitFileInput(PyFileInputTree pyFileInputTree) {
    scan(pyFileInputTree.statements());
  }

  @Override
  public void visitStatementList(PyStatementListTree pyStatementListTree) {
    scan(pyStatementListTree.statements());
  }

  @Override
  public void visitIfStatement(PyIfStatementTree pyIfStatementTree) {
    scan(pyIfStatementTree.condition());
    scan(pyIfStatementTree.body());
    scan(pyIfStatementTree.elifBranches());
    scan(pyIfStatementTree.elseBranch());
  }

  @Override
  public void visitElseStatement(PyElseStatementTree pyElseStatementTree) {
    scan(pyElseStatementTree.body());
  }

  @Override
  public void visitExecStatement(PyExecStatementTree pyExecStatementTree) {
    scan(pyExecStatementTree.expression());
    scan(pyExecStatementTree.globalsExpression());
    scan(pyExecStatementTree.localsExpression());
  }

  @Override
  public void visitAssertStatement(PyAssertStatementTree pyAssertStatementTree) {
    scan(pyAssertStatementTree.expressions());
  }

  @Override
  public void visitDelStatement(PyDelStatementTree pyDelStatementTree) {
    scan(pyDelStatementTree.expressions());
  }

  @Override
  public void visitPassStatement(PyPassStatementTree pyPassStatementTree) {
    // nothing to visit for pass statement
  }

  @Override
  public void visitPrintStatement(PyPrintStatementTree pyPrintStatementTree) {
    scan(pyPrintStatementTree.expressions());
  }

  @Override
  public void visitReturnStatement(PyReturnStatementTree pyReturnStatementTree) {
    scan(pyReturnStatementTree.expressions());
  }

  @Override
  public void visitYieldStatement(PyYieldStatementTree pyYieldStatementTree) {
    scan(pyYieldStatementTree.yieldExpression());
  }

  @Override
  public void visitYieldExpression(PyYieldExpressionTree pyYieldExpressionTree) {
    scan(pyYieldExpressionTree.expressions());
  }

  @Override
  public void visitRaiseStatement(PyRaiseStatementTree pyRaiseStatementTree) {
    scan(pyRaiseStatementTree.expressions());
    scan(pyRaiseStatementTree.fromExpression());
  }

  @Override
  public void visitBreakStatement(PyBreakStatementTree pyBreakStatementTree) {
    // nothing to visit for break statement
  }

  @Override
  public void visitContinueStatement(PyContinueStatementTree pyContinueStatementTree) {
    // nothing to visit for continue statement
  }

  @Override
  public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
    scan(pyFunctionDefTree.decorators());
    scan(pyFunctionDefTree.name());
    scan(pyFunctionDefTree.parameters());
    scan(pyFunctionDefTree.returnTypeAnnotation());
    scan(pyFunctionDefTree.body());
  }

  @Override
  public void visitName(PyNameTree pyNameTree) {
    // nothing to scan on a name
  }

  @Override
  public void visitClassDef(PyClassDefTree pyClassDefTree) {
    scan(pyClassDefTree.name());
    scan(pyClassDefTree.args());
    scan(pyClassDefTree.body());
  }

  @Override
  public void visitAliasedName(PyAliasedNameTree pyAliasedNameTree) {
    scan(pyAliasedNameTree.dottedName());
    scan(pyAliasedNameTree.alias());
  }

  @Override
  public void visitDottedName(PyDottedNameTree pyDottedNameTree) {
    scan(pyDottedNameTree.names());
  }

  @Override
  public void visitImportFrom(PyImportFromTree pyImportFromTree) {
    scan(pyImportFromTree.module());
    scan(pyImportFromTree.importedNames());
  }

  @Override
  public void visitForStatement(PyForStatementTree pyForStatementTree) {
    scan(pyForStatementTree.expressions());
    scan(pyForStatementTree.testExpressions());
    scan(pyForStatementTree.body());
    scan(pyForStatementTree.elseBody());
  }

  @Override
  public void visitImportName(PyImportNameTree pyImportNameTree) {
    scan(pyImportNameTree.modules());
  }

  @Override
  public void visitGlobalStatement(PyGlobalStatementTree pyGlobalStatementTree) {
    scan(pyGlobalStatementTree.variables());
  }

  @Override
  public void visitNonlocalStatement(PyNonlocalStatementTree pyNonlocalStatementTree) {
    scan(pyNonlocalStatementTree.variables());
  }

  @Override
  public void visitWhileStatement(PyWhileStatementTree pyWhileStatementTree) {
    scan(pyWhileStatementTree.condition());
    scan(pyWhileStatementTree.body());
    scan(pyWhileStatementTree.elseBody());
  }

  @Override
  public void visitExpressionStatement(PyExpressionStatementTree pyExpressionStatementTree) {
    scan(pyExpressionStatementTree.expressions());
  }

  @Override
  public void visitTryStatement(PyTryStatementTree pyTryStatementTree) {
    scan(pyTryStatementTree.body());
    scan(pyTryStatementTree.exceptClauses());
    scan(pyTryStatementTree.finallyClause());
    scan(pyTryStatementTree.elseClause());
  }

  @Override
  public void visitFinallyClause(PyFinallyClauseTree pyFinallyClauseTree) {
    scan(pyFinallyClauseTree.body());
  }

  @Override
  public void visitExceptClause(PyExceptClauseTree pyExceptClauseTree) {
    scan(pyExceptClauseTree.exception());
    scan(pyExceptClauseTree.exceptionInstance());
    scan(pyExceptClauseTree.body());
  }

  @Override
  public void visitWithStatement(PyWithStatementTree pyWithStatementTree) {
    scan(pyWithStatementTree.withItems());
    scan(pyWithStatementTree.statements());
  }

  @Override
  public void visitWithItem(PyWithItemTree pyWithItemTree) {
    scan(pyWithItemTree.test());
    scan(pyWithItemTree.expression());
  }

  @Override
  public void visitQualifiedExpression(PyQualifiedExpressionTree pyQualifiedExpressionTree) {
    scan(pyQualifiedExpressionTree.qualifier());
    scan(pyQualifiedExpressionTree.name());
  }

  @Override
  public void visitCallExpression(PyCallExpressionTree pyCallExpressionTree) {
    scan(pyCallExpressionTree.callee());
    scan(pyCallExpressionTree.arguments());
  }

  @Override
  public void visitArgumentList(PyArgListTree pyArgListTree) {
    scan(pyArgListTree.arguments());
  }

  @Override
  public void visitArgument(PyArgumentTree pyArgumentTree) {
    scan(pyArgumentTree.keywordArgument());
    scan(pyArgumentTree.expression());
  }

  @Override
  public void visitAssignmentStatement(PyAssignmentStatementTree pyAssignmentStatementTree) {
    scan(pyAssignmentStatementTree.lhsExpressions());
    scan(pyAssignmentStatementTree.assignedValue());
  }

  @Override
  public void visitExpressionList(PyExpressionListTree pyExpressionListTree) {
    scan(pyExpressionListTree.expressions());
  }

  @Override
  public void visitBinaryExpression(PyBinaryExpressionTree pyBinaryExpressionTree) {
    scan(pyBinaryExpressionTree.leftOperand());
    scan(pyBinaryExpressionTree.rightOperand());
  }

  @Override
  public void visitLambda(PyLambdaExpressionTree pyLambdaExpressionTree) {
    scan(pyLambdaExpressionTree.parameters());
    scan(pyLambdaExpressionTree.expression());
  }

  @Override
  public void visitParameterList(PyParameterListTree pyParameterListTree) {
    scan(pyParameterListTree.all());
  }

  @Override
  public void visitTupleParameter(PyTupleParameterTree tree) {
    scan(tree.parameters());
  }

  @Override
  public void visitParameter(PyParameterTree tree) {
    scan(tree.name());
    scan(tree.typeAnnotation());
    scan(tree.defaultValue());
  }

  @Override
  public void visitTypeAnnotation(PyTypeAnnotationTree tree) {
    scan(tree.expression());
  }

  @Override
  public void visitNumericLiteral(PyNumericLiteralTree pyNumericLiteralTree) {
    // noop
  }

  @Override
  public void visitStringLiteral(PyStringLiteralTree pyStringLiteralTree) {
    scan(pyStringLiteralTree.stringElements());
  }

  @Override
  public void visitStringElement(PyStringElementTree tree) {
    // noop
  }

  @Override
  public void visitListLiteral(PyListLiteralTree pyListLiteralTree) {
    scan(pyListLiteralTree.elements());
  }

  @Override
  public void visitUnaryExpression(PyUnaryExpressionTree pyUnaryExpressionTree) {
    scan(pyUnaryExpressionTree.expression());
  }

  @Override
  public void visitStarredExpression(PyStarredExpressionTree pyStarredExpressionTree) {
    scan(pyStarredExpressionTree.expression());
  }

  @Override
  public void visitAwaitExpression(PyAwaitExpressionTree pyAwaitExpressionTree) {
    scan(pyAwaitExpressionTree.expression());
  }

  @Override
  public void visitSliceExpression(PySliceExpressionTree pySliceExpressionTree) {
    scan(pySliceExpressionTree.object());
    scan(pySliceExpressionTree.sliceList());
  }

  @Override
  public void visitSliceList(PySliceListTree pySliceListTree) {
    scan(pySliceListTree.slices());
  }

  @Override
  public void visitSliceItem(PySliceItemTree pySliceItemTree) {
    scan(pySliceItemTree.lowerBound());
    scan(pySliceItemTree.upperBound());
    scan(pySliceItemTree.stride());
  }

  @Override
  public void visitSubscriptionExpression(PySubscriptionExpressionTree pySubscriptionExpressionTree) {
    scan(pySubscriptionExpressionTree.object());
    scan(pySubscriptionExpressionTree.subscripts());
  }

  @Override
  public void visitParenthesizedExpression(PyParenthesizedExpressionTree pyParenthesizedExpressionTree) {
    scan(pyParenthesizedExpressionTree.expression());
  }

  @Override
  public void visitTuple(PyTupleTree pyTupleTree) {
    scan(pyTupleTree.elements());
  }

  @Override
  public void visitConditionalExpression(PyConditionalExpressionTree pyConditionalExpressionTree) {
    scan(pyConditionalExpressionTree.condition());
    scan(pyConditionalExpressionTree.trueExpression());
    scan(pyConditionalExpressionTree.falseExpression());
  }

  @Override
  public void visitPyListOrSetCompExpression(PyComprehensionExpressionTree tree) {
    scan(tree.resultExpression());
    scan(tree.comprehensionFor());
  }

  @Override
  public void visitComprehensionFor(PyComprehensionForTree tree) {
    scan(tree.loopExpression());
    scan(tree.iterable());
    scan(tree.nestedClause());
  }

  @Override
  public void visitComprehensionIf(PyComprehensionIfTree tree) {
    scan(tree.condition());
    scan(tree.nestedClause());
  }

  @Override
  public void visitDictionaryLiteral(PyDictionaryLiteralTree pyDictionaryLiteralTree) {
    scan(pyDictionaryLiteralTree.elements());
  }

  @Override
  public void visitSetLiteral(PySetLiteralTree pySetLiteralTree) {
    scan((pySetLiteralTree.elements()));
  }

  @Override
  public void visitKeyValuePair(PyKeyValuePairTree pyKeyValuePairTree) {
    scan(pyKeyValuePairTree.expression());
    scan(pyKeyValuePairTree.key());
    scan(pyKeyValuePairTree.value());
  }

  @Override
  public void visitDictCompExpression(PyDictCompExpressionTreeImpl tree) {
    scan(tree.keyExpression());
    scan(tree.valueExpression());
    scan(tree.comprehensionFor());
  }

  @Override
  public void visitCompoundAssignment(PyCompoundAssignmentStatementTree pyCompoundAssignmentStatementTree) {
    scan(pyCompoundAssignmentStatementTree.lhsExpression());
    scan(pyCompoundAssignmentStatementTree.rhsExpression());
  }

  @Override
  public void visitAnnotatedAssignment(PyAnnotatedAssignmentTree pyAnnotatedAssignmentTree) {
    scan(pyAnnotatedAssignmentTree.variable());
    scan(pyAnnotatedAssignmentTree.annotation());
    scan(pyAnnotatedAssignmentTree.assignedValue());
  }

  @Override
  public void visitNone(PyNoneExpressionTree pyNoneExpressionTree) {
    // noop
  }

  @Override
  public void visitRepr(PyReprExpressionTree pyReprExpressionTree) {
    scan(pyReprExpressionTree.expressionList());
  }

  @Override
  public void visitEllipsis(PyEllipsisExpressionTree pyEllipsisExpressionTree) {
    // noop
  }

  @Override
  public void visitDecorator(PyDecoratorTree pyDecoratorTree) {
    scan(pyDecoratorTree.name());
    scan(pyDecoratorTree.arguments());
  }
}
