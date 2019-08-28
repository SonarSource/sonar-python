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
package org.sonar.python.api.tree;

public interface PyTreeVisitor {

  void visitFileInput(PyFileInputTree pyFileInputTree);

  void visitStatementList(PyStatementListTree pyStatementListTree);

  void visitIfStatement(PyIfStatementTree pyIfStatementTree);

  void visitElseStatement(PyElseStatementTree pyElseStatementTree);

  void visitExecStatement(PyExecStatementTree pyExecStatementTree);

  void visitAssertStatement(PyAssertStatementTree pyAssertStatementTree);

  void visitDelStatement(PyDelStatementTree pyDelStatementTree);

  void visitPassStatement(PyPassStatementTree pyPassStatementTree);

  void visitPrintStatement(PyPrintStatementTree pyPrintStatementTree);

  void visitReturnStatement(PyReturnStatementTree pyReturnStatementTree);

  void visitYieldStatement(PyYieldStatementTree pyYieldStatementTree);

  void visitYieldExpression(PyYieldExpressionTree pyYieldExpressionTree);

  void visitRaiseStatement(PyRaiseStatementTree pyRaiseStatementTree);

  void visitBreakStatement(PyBreakStatementTree pyBreakStatementTree);

  void visitContinueStatement(PyContinueStatementTree pyContinueStatementTree);

  void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree);

  void visitName(PyNameTree pyNameTree);

  void visitClassDef(PyClassDefTree pyClassDefTree);

  void visitAliasedName(PyAliasedNameTree pyAliasedNameTree);

  void visitDottedName(PyDottedNameTree pyDottedNameTree);

  void visitImportFrom(PyImportFromTree pyImportFromTree);

  void visitImportName(PyImportNameTree pyImportNameTree);

  void visitForStatement(PyForStatementTree pyForStatementTree);

  void visitGlobalStatement(PyGlobalStatementTree pyGlobalStatementTree);

  void visitNonlocalStatement(PyNonlocalStatementTree pyNonlocalStatementTree);

  void visitWhileStatement(PyWhileStatementTree pyWhileStatementTree);

  void visitExpressionStatement(PyExpressionStatementTree pyExpressionStatementTree);

  void visitTryStatement(PyTryStatementTree pyTryStatementTree);

  void visitFinallyClause(PyFinallyClauseTree pyFinallyClauseTree);

  void visitExceptClause(PyExceptClauseTree pyExceptClauseTree);

  void visitWithStatement(PyWithStatementTree pyWithStatementTree);

  void visitWithItem(PyWithItemTree pyWithItemTree);

  void visitQualifiedExpression(PyQualifiedExpressionTree pyQualifiedExpressionTree);

  void visitCallExpression(PyCallExpressionTree pyCallExpressionTree);

  void visitArgument(PyArgumentTree pyArgumentTree);

  void visitAssignmentStatement(PyAssignmentStatementTree pyAssignmentStatementTree);

  void visitExpressionList(PyExpressionListTree pyExpressionListTree);

  void visitBinaryExpression(PyBinaryExpressionTree pyBinaryExpressionTree);

  void visitLambda(PyLambdaExpressionTree pyLambdaExpressionTree);

  void visitTypedArgList(PyTypedArgListTree pyTypedArgListTree);

  void visitTypeArgument(PyTypedArgumentTree pyTypedArgumentTree);

  void visitNumericLiteral(PyNumericLiteralTree pyNumericLiteralTree);

  void visitListLiteral(PyListLiteralTree pyListLiteralTree);

  void visitUnaryExpression(PyUnaryExpressionTree pyUnaryExpressionTree);

  void visitStringLiteral(PyStringLiteralTree pyStringLiteralTree);
}
