/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
  MATRIX_MULT_ASSIGN("@=")

  ;

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
