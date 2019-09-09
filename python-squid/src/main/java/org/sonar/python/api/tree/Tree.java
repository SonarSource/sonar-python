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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import java.util.List;
import javax.annotation.CheckForNull;

public interface Tree {

  void accept(PyTreeVisitor visitor);

  boolean is(Kind kind);

  @CheckForNull
  @Deprecated
  AstNode astNode();

  Token firstToken();

  Token lastToken();

  Tree parent();

  List<Tree> children();

  enum Kind {
    ALIASED_NAME(PyAliasedNameTree.class),

    ARGUMENT(PyArgumentTree.class),

    ARG_LIST(PyArgListTree.class),

    ASSERT_STMT(PyAssertStatementTree.class),

    ASSIGNMENT_STMT(PyAssignmentStatementTree.class),

    BREAK_STMT(PyBreakStatementTree.class),

    CALL_EXPR(PyCallExpressionTree.class),

    CLASSDEF(PyClassDefTree.class),

    CONDITIONAL_EXPR(PyConditionalExpressionTree.class),

    CONTINUE_STMT(PyContinueStatementTree.class),

    COMPOUND_ASSIGNMENT(PyCompoundAssignmentStatementTree.class),

    DICTIONARY_LITERAL(PyDictionaryLiteralTree.class),

    DEL_STMT(PyDelStatementTree.class),

    DOTTED_NAME(PyDottedNameTree.class),

    ELSE_STMT(PyElseStatementTree.class),

    EXCEPT_CLAUSE(PyExceptClauseTree.class),

    EXEC_STMT(PyExecStatementTree.class),

    EXPRESSION_LIST(PyExpressionListTree.class),

    EXPRESSION_STMT(PyExpressionStatementTree.class),

    FILE_INPUT(PyFileInputTree.class),

    FINALLY_CLAUSE(PyFinallyClauseTree.class),

    FOR_STMT(PyForStatementTree.class),

    FUNCDEF(PyFunctionDefTree.class),

    GLOBAL_STMT(PyGlobalStatementTree.class),

    IF_STMT(PyIfStatementTree.class),

    IMPORT_FROM(PyImportFromTree.class),

    IMPORT_NAME(PyDottedNameTree.class),

    IMPORT_STMT(PyDottedNameTree.class),

    LAMBDA(PyLambdaExpressionTree.class),

    LIST_LITERAL(PyListLiteralTree.class),

    NAME(PyNameTree.class),

    NONLOCAL_STMT(PyNonlocalStatementTree.class),

    NUMERIC_LITERAL(PyNumericLiteralTree.class),

    PASS_STMT(PyPassStatementTree.class),

    PRINT_STMT(PyPrintStatementTree.class),

    QUALIFIED_EXPR(PyQualifiedExpressionTree.class),

    RAISE_STMT(PyRaiseStatementTree.class),

    RETURN_STMT(PyReturnStatementTree.class),

    SET_LITERAL(PySetLiteralTree.class),

    STATEMENT_LIST(PyStatementListTree.class),

    STRING_LITERAL(PyStringLiteralTree.class),

    STRING_ELEMENT(PyStringElementTree.class),

    TRY_STMT(PyTryStatementTree.class),

    TYPED_ARG(PyTypedArgumentTree.class),

    TYPED_ARG_LIST(PyTypedArgListTree.class),

    WHILE_STMT(PyWhileStatementTree.class),

    WITH_ITEM(PyWithItemTree.class),

    WITH_STMT(PyWithStatementTree.class),

    YIELD_EXPR(PyYieldExpressionTree.class),

    YIELD_STMT(PyYieldStatementTree.class),

    PARENTHESIZED(PyParenthesizedExpressionTree.class),

    STARRED_EXPR(PyStarredExpressionTree.class),

    AWAIT(PyAwaitExpressionTree.class),

    TUPLE(PyTupleTree.class),

    DICT_COMPREHENSION(PyDictCompExpressionTree.class),
    LIST_COMPREHENSION(PyListOrSetCompExpressionTree.class),
    SET_COMPREHENSION(PyListOrSetCompExpressionTree.class),
    COMP_FOR(PyComprehensionForTree.class),
    COMP_IF(PyComprehensionIfTree.class),

    SUBSCRIPTION(PySubscriptionExpressionTree.class),

    SLICE_EXPR(PySliceExpressionTree.class),
    SLICE_LIST(PySliceListTree.class),
    SLICE_ITEM(PySliceItemTree.class),

    PLUS(PyBinaryExpressionTree.class),
    MINUS(PyBinaryExpressionTree.class),
    MULTIPLICATION(PyBinaryExpressionTree.class),
    DIVISION(PyBinaryExpressionTree.class),
    FLOOR_DIVISION(PyBinaryExpressionTree.class),
    MODULO(PyBinaryExpressionTree.class),
    MATRIX_MULTIPLICATION(PyBinaryExpressionTree.class),
    SHIFT_EXPR(PyBinaryExpressionTree.class),
    BITWISE_AND(PyBinaryExpressionTree.class),
    BITWISE_OR(PyBinaryExpressionTree.class),
    BITWISE_XOR(PyBinaryExpressionTree.class),
    AND(PyBinaryExpressionTree.class),
    OR(PyBinaryExpressionTree.class),
    COMPARISON(PyBinaryExpressionTree.class),
    POWER(PyBinaryExpressionTree.class),
    IN(PyInExpressionTree.class),
    IS(PyIsExpressionTree.class),

    UNARY_PLUS(PyUnaryExpressionTree.class),
    UNARY_MINUS(PyUnaryExpressionTree.class),
    BITWISE_COMPLEMENT(PyUnaryExpressionTree.class),
    NOT(PyUnaryExpressionTree.class),

    KEY_VALUE_PAIR(PyKeyValuePairTree.class);

    final Class<? extends Tree> associatedInterface;

    Kind(Class<? extends Tree> associatedInterface) {
      this.associatedInterface = associatedInterface;
    }
  }

  Kind getKind();
}
