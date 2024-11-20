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
package org.sonar.python.api;

import org.sonar.sslr.grammar.GrammarRuleKey;

public enum PythonGrammar implements GrammarRuleKey {
  FACTOR,
  TRAILER,
  SUBSCRIPTLIST,
  SUBSCRIPT,
  SLICEOP,
  TESTLIST_COMP,
  DICTORSETMAKER,

  ARGLIST,
  ARGUMENT,

  NAME,

  VARARGSLIST,
  FPDEF,
  FPLIST,

  TYPEDARGSLIST,
  TFPDEF,
  TFPLIST,

  TEST,
  TESTLIST,

  COMP_FOR,
  COMP_ITER,
  COMP_IF,
  TEST_NOCOND,
  EXPRLIST,
  EXPR,
  STAR_EXPR,

  TESTLIST_STAR_EXPR,

  YIELD_EXPR,

  // Expressions

  ATOM,

  POWER,

  A_EXPR,
  M_EXPR,

  SHIFT_EXPR,

  XOR_EXPR,
  AND_EXPR,
  OR_EXPR,

  NAMED_EXPR_TEST,
  STAR_NAMED_EXPRESSIONS,
  STAR_NAMED_EXPRESSION,
  FSTRING_REPLACEMENT_FIELD,
  FSTRING,
  FORMAT_SPECIFIER,
  STRINGS,

  COMPARISON,
  COMP_OPERATOR,

  OR_TEST,
  AND_TEST,
  NOT_TEST,

  LAMBDEF,
  LAMBDEF_NOCOND,

  ELLIPSIS,

  // Simple statements

  SIMPLE_STMT,
  EXPRESSION_STMT,
  PRINT_STMT,
  EXEC_STMT,
  ASSERT_STMT,

  ANNASSIGN,
  AUGASSIGN,
  ANNOTATED_RHS,

  PASS_STMT,
  DEL_STMT,
  RETURN_STMT,
  YIELD_STMT,
  RAISE_STMT,
  BREAK_STMT,
  CONTINUE_STMT,

  IMPORT_STMT,
  IMPORT_NAME,
  IMPORT_FROM,
  IMPORT_AS_NAME,
  DOTTED_AS_NAME,
  IMPORT_AS_NAMES,
  DOTTED_AS_NAMES,

  GLOBAL_STMT,
  NONLOCAL_STMT,

  // Compound statements

  COMPOUND_STMT,
  SUITE,
  STATEMENT,
  STMT_LIST,

  IF_STMT,
  WHILE_STMT,
  FOR_STMT,

  TRY_STMT,
  EXCEPT_CLAUSE,

  WITH_STMT,
  WITH_ITEM,

  MATCH_STMT,
  SUBJECT_EXPR,
  CASE_BLOCK,
  GUARD,

  PATTERNS,
  PATTERN,
  AS_PATTERN,
  OR_PATTERN,
  CLOSED_PATTERN,
  LITERAL_PATTERN,
  CAPTURE_PATTERN,
  SEQUENCE_PATTERN,
  STAR_PATTERN,
  MAYBE_STAR_PATTERN,
  MAYBE_SEQUENCE_PATTERN,
  OPEN_SEQUENCE_PATTERN,
  WILDCARD_PATTERN,
  GROUP_PATTERN,
  CLASS_PATTERN,
  PATTERN_ARGS,
  PATTERN_ARG,
  KEYWORD_PATTERN,
  NAME_OR_ATTR,
  VALUE_PATTERN,
  MAPPING_PATTERN,
  ITEMS_PATTERN,
  KEY_VALUE_PATTERN,
  DOUBLE_STAR_PATTERN,

  SIGNED_NUMBER,
  COMPLEX_NUMBER,

  FUNCDEF,
  DECORATORS,
  DECORATOR,
  DOTTED_NAME,
  ATTR,
  FUNCNAME,
  FUN_RETURN_ANNOTATION,
  TYPE_PARAMS,
  TYPE_PARAM,
  TYPE_ALIAS_STMT,
  TYPE_ANNOTATION,

  CLASSDEF,
  CLASSNAME,

  ASYNC_STMT,

  // Top-level components
  FILE_INPUT;

}
