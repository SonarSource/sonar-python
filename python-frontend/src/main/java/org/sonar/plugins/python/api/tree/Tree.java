/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.plugins.python.api.tree;

import java.util.List;

public interface Tree {

  void accept(TreeVisitor visitor);

  boolean is(Kind... kinds);

  Token firstToken();

  /**
   * @return the last meaningful token of the Tree.
   * Separators of simple statements (semicolon and/or newline) are not be returned by this method.
   */
  Token lastToken();

  Tree parent();

  List<Tree> children();

  enum Kind {
    ALIASED_NAME(AliasedName.class),

    REGULAR_ARGUMENT(RegularArgument.class),

    ARG_LIST(ArgList.class),

    ANNOTATED_ASSIGNMENT(AnnotatedAssignment.class),

    ASSERT_STMT(AssertStatement.class),

    ASSIGNMENT_STMT(AssignmentStatement.class),

    BREAK_STMT(BreakStatement.class),

    CALL_EXPR(CallExpression.class),

    CLASSDEF(ClassDef.class),

    CONDITIONAL_EXPR(ConditionalExpression.class),

    CONTINUE_STMT(ContinueStatement.class),

    COMPOUND_ASSIGNMENT(CompoundAssignmentStatement.class),

    DICTIONARY_LITERAL(DictionaryLiteral.class),

    DECORATOR(Decorator.class),

    DEL_STMT(DelStatement.class),

    DOTTED_NAME(DottedName.class),

    ELLIPSIS(EllipsisExpression.class),

    ELSE_CLAUSE(ElseClause.class),

    EXCEPT_CLAUSE(ExceptClause.class),

    EXEC_STMT(ExecStatement.class),

    EXPRESSION_LIST(ExpressionList.class),

    EXPRESSION_STMT(ExpressionStatement.class),

    FILE_INPUT(FileInput.class),

    FINALLY_CLAUSE(FinallyClause.class),

    FOR_STMT(ForStatement.class),

    FUNCDEF(FunctionDef.class),

    GLOBAL_STMT(GlobalStatement.class),

    IF_STMT(IfStatement.class),

    IMPORT_FROM(ImportFrom.class),

    IMPORT_NAME(DottedName.class),

    IMPORT_STMT(DottedName.class),

    LAMBDA(LambdaExpression.class),

    LIST_LITERAL(ListLiteral.class),

    NAME(Name.class),

    NONLOCAL_STMT(NonlocalStatement.class),

    NONE(NoneExpression.class),

    NUMERIC_LITERAL(NumericLiteral.class),

    PASS_STMT(PassStatement.class),

    PRINT_STMT(PrintStatement.class),

    QUALIFIED_EXPR(QualifiedExpression.class),

    RAISE_STMT(RaiseStatement.class),

    REPR(ReprExpression.class),

    RETURN_STMT(ReturnStatement.class),

    SET_LITERAL(SetLiteral.class),

    STATEMENT_LIST(StatementList.class),

    STRING_LITERAL(StringLiteral.class),

    STRING_ELEMENT(StringElement.class),

    FORMATTED_EXPRESSION(FormattedExpression.class),

    FORMAT_SPECIFIER(FormatSpecifier.class),

    TRY_STMT(TryStatement.class),

    PARAMETER(Parameter.class),
    TUPLE_PARAMETER(TupleParameter.class),

    VARIABLE_TYPE_ANNOTATION(TypeAnnotation.class),
    PARAMETER_TYPE_ANNOTATION(TypeAnnotation.class),
    RETURN_TYPE_ANNOTATION(TypeAnnotation.class),

    PARAMETER_LIST(ParameterList.class),

    WHILE_STMT(WhileStatement.class),

    WITH_ITEM(WithItem.class),

    WITH_STMT(WithStatement.class),

    YIELD_EXPR(YieldExpression.class),

    YIELD_STMT(YieldStatement.class),

    PARENTHESIZED(ParenthesizedExpression.class),

    UNPACKING_EXPR(UnpackingExpression.class),

    AWAIT(AwaitExpression.class),

    TUPLE(Tuple.class),

    DICT_COMPREHENSION(DictCompExpression.class),
    LIST_COMPREHENSION(ComprehensionExpression.class),
    SET_COMPREHENSION(ComprehensionExpression.class),
    GENERATOR_EXPR(ComprehensionExpression.class),
    COMP_FOR(ComprehensionFor.class),
    COMP_IF(ComprehensionIf.class),

    SUBSCRIPTION(SubscriptionExpression.class),

    SLICE_EXPR(SliceExpression.class),
    SLICE_LIST(SliceList.class),
    SLICE_ITEM(SliceItem.class),

    PLUS(BinaryExpression.class),
    MINUS(BinaryExpression.class),
    MULTIPLICATION(BinaryExpression.class),
    DIVISION(BinaryExpression.class),
    FLOOR_DIVISION(BinaryExpression.class),
    MODULO(BinaryExpression.class),
    MATRIX_MULTIPLICATION(BinaryExpression.class),
    SHIFT_EXPR(BinaryExpression.class),
    BITWISE_AND(BinaryExpression.class),
    BITWISE_OR(BinaryExpression.class),
    BITWISE_XOR(BinaryExpression.class),
    AND(BinaryExpression.class),
    OR(BinaryExpression.class),
    COMPARISON(BinaryExpression.class),
    POWER(BinaryExpression.class),
    IN(InExpression.class),
    IS(IsExpression.class),

    UNARY_PLUS(UnaryExpression.class),
    UNARY_MINUS(UnaryExpression.class),
    BITWISE_COMPLEMENT(UnaryExpression.class),
    NOT(UnaryExpression.class),

    ASSIGNMENT_EXPRESSION(AssignmentExpression.class),

    KEY_VALUE_PAIR(KeyValuePair.class),
    TOKEN(Token.class);
    final Class<? extends Tree> associatedInterface;

    Kind(Class<? extends Tree> associatedInterface) {
      this.associatedInterface = associatedInterface;
    }
  }

  Kind getKind();

}
