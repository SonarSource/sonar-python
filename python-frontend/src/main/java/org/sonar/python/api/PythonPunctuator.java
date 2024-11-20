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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.TokenType;

public enum PythonPunctuator implements TokenType {

  // Operators

  PLUS("+"),
  MINUS("-"),
  MUL("*"),
  MUL_MUL("**"),
  DIV("/"),
  DIV_DIV("//"),
  MOD("%"),
  LEFT_OP("<<"),
  RIGHT_OP(">>"),

  /**
   * Bitwise AND.
   */
  AND("&"),

  /**
   * Bitwise OR.
   */
  OR("|"),
  XOR("^"),
  TILDE("~"),
  LT("<"),
  GT(">"),
  LT_EQU("<="),
  GT_EQU(">="),
  EQU("=="),
  NOT_EQU("!="),
  NOT_EQU2("<>"),

  // Delimiters

  BACKTICK("`"),
  LPARENTHESIS("("),
  RPARENTHESIS(")"),
  LBRACKET("["),
  RBRACKET("]"),
  LCURLYBRACE("{"),
  RCURLYBRACE("}"),
  COMMA(","),
  COLON(":"),
  DOT("."),
  SEMICOLON(";"),
  AT("@"),
  ASSIGN("="),
  PLUS_ASSIGN("+="),
  MINUS_ASSIGN("-="),
  MUL_ASSIGN("*="),
  DIV_ASSIGN("/="),
  DIV_DIV_ASSIGN("//="),
  MOD_ASSIGN("%="),
  AND_ASSIGN("&="),
  OR_ASSIGN("|="),
  XOR_ASSIGN("^="),
  RIGHT_ASSIGN(">>="),
  LEFT_ASSIGN("<<="),
  MUL_MUL_ASSIGN("**="),
  MATRIX_MULT_ASSIGN("@="),
  WALRUS_OPERATOR(":=");

  private final String value;

  PythonPunctuator(String word) {
    this.value = word;
  }

  @Override
  public String getName() {
    return name();
  }

  @Override
  public String getValue() {
    return value;
  }

  @Override
  public boolean hasToBeSkippedFromAst(AstNode node) {
    return false;
  }

}
