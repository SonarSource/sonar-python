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

    AS_PATTERN(AsPattern.class),

    REGULAR_ARGUMENT(RegularArgument.class),

    ARG_LIST(ArgList.class),

    ANNOTATED_ASSIGNMENT(AnnotatedAssignment.class),

    ASSERT_STMT(AssertStatement.class),

    ASSIGNMENT_STMT(AssignmentStatement.class),

    BREAK_STMT(BreakStatement.class),

    BOOLEAN_LITERAL_PATTERN(LiteralPattern.class),

    CALL_EXPR(CallExpression.class),

    CAPTURE_PATTERN(CapturePattern.class),

    CASE_BLOCK(CaseBlock.class),

    CLASSDEF(ClassDef.class),

    CLASS_PATTERN(ClassPattern.class),

    CONDITIONAL_EXPR(ConditionalExpression.class),

    CONTINUE_STMT(ContinueStatement.class),

    COMPOUND_ASSIGNMENT(CompoundAssignmentStatement.class),

    DICTIONARY_LITERAL(DictionaryLiteral.class),

    DECORATOR(Decorator.class),

    DEL_STMT(DelStatement.class),

    DOTTED_NAME(DottedName.class),

    DOUBLE_STAR_PATTERN(DoubleStarPattern.class),

    ELLIPSIS(EllipsisExpression.class),

    ELSE_CLAUSE(ElseClause.class),

    EXCEPT_CLAUSE(ExceptClause.class),

    EXCEPT_GROUP_CLAUSE(ExceptClause.class),

    EXEC_STMT(ExecStatement.class),

    EXPRESSION_LIST(ExpressionList.class),

    EXPRESSION_STMT(ExpressionStatement.class),

    FILE_INPUT(FileInput.class),

    FINALLY_CLAUSE(FinallyClause.class),

    FOR_STMT(ForStatement.class),

    FUNCDEF(FunctionDef.class),

    GLOBAL_STMT(GlobalStatement.class),

    GROUP_PATTERN(GroupPattern.class),

    GUARD(Guard.class),

    IF_STMT(IfStatement.class),

    IMPORT_FROM(ImportFrom.class),

    IMPORT_NAME(DottedName.class),

    KEYWORD_PATTERN(KeywordPattern.class),

    LAMBDA(LambdaExpression.class),

    LIST_LITERAL(ListLiteral.class),

    MATCH_STMT(MatchStatement.class),

    MAPPING_PATTERN(MappingPattern.class),

    NAME(Name.class),

    NONLOCAL_STMT(NonlocalStatement.class),

    NONE(NoneExpression.class),

    NONE_LITERAL_PATTERN(LiteralPattern.class),

    NUMERIC_LITERAL(NumericLiteral.class),

    NUMERIC_LITERAL_PATTERN(LiteralPattern.class),

    OR_PATTERN(OrPattern.class),

    PASS_STMT(PassStatement.class),

    PRINT_STMT(PrintStatement.class),

    QUALIFIED_EXPR(QualifiedExpression.class),

    RAISE_STMT(RaiseStatement.class),

    REPR(ReprExpression.class),

    RETURN_STMT(ReturnStatement.class),

    SEQUENCE_PATTERN(SequencePattern.class),

    SET_LITERAL(SetLiteral.class),

    STATEMENT_LIST(StatementList.class),

    STAR_PATTERN(StarPattern.class),

    STRING_LITERAL(StringLiteral.class),

    STRING_LITERAL_PATTERN(LiteralPattern.class),

    STRING_ELEMENT(StringElement.class),

    FORMATTED_EXPRESSION(FormattedExpression.class),

    FORMAT_SPECIFIER(FormatSpecifier.class),

    TRY_STMT(TryStatement.class),

    PARAMETER(Parameter.class),
    TUPLE_PARAMETER(TupleParameter.class),

    VARIABLE_TYPE_ANNOTATION(TypeAnnotation.class),
    PARAMETER_TYPE_ANNOTATION(TypeAnnotation.class),
    TYPE_PARAM_TYPE_ANNOTATION(TypeAnnotation.class),
    RETURN_TYPE_ANNOTATION(TypeAnnotation.class),
    TYPE_PARAMS(TypeParams.class),
    TYPE_PARAM(TypeParam.class),

    PARAMETER_LIST(ParameterList.class),
    VALUE_PATTERN(ValuePattern.class),

    WHILE_STMT(WhileStatement.class),

    WILDCARD_PATTERN(WildcardPattern.class),

    WITH_ITEM(WithItem.class),

    WITH_STMT(WithStatement.class),

    YIELD_EXPR(YieldExpression.class),

    YIELD_STMT(YieldStatement.class),

    TYPE_ALIAS_STMT(TypeAliasStatement.class),

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
    KEY_VALUE_PATTERN(KeyValuePattern.class),
    TOKEN(Token.class),
    LINE_MAGIC(LineMagic.class),
    DYNAMIC_OBJECT_INFO_STATEMENT(DynamicObjectInfoStatement.class),
    CELL_MAGIC_STATEMENT(CellMagicStatement.class);

    final Class<? extends Tree> associatedInterface;

    Kind(Class<? extends Tree> associatedInterface) {
      this.associatedInterface = associatedInterface;
    }
  }

  Kind getKind();

}
