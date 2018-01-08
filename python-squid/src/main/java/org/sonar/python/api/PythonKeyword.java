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

/**
 * http://docs.python.org/reference/lexical_analysis.html#keywords
 */
public enum PythonKeyword implements TokenType {

  NONE("None"),

  /**
   * Logical AND.
   */
  AND("and"),

  AS("as"),
  ASSERT("assert"),
  BREAK("break"),
  CLASS("class"),
  CONTINUE("continue"),
  DEF("def"),
  DEL("del"),
  ELIF("elif"),
  ELSE("else"),
  EXCEPT("except"),
  FINALLY("finally"),
  FOR("for"),
  FROM("from"),
  GLOBAL("global"),
  IF("if"),
  IMPORT("import"),
  IN("in"),
  IS("is"),
  LAMBDA("lambda"),
  NONLOCAL("nonlocal"),
  NOT("not"),

  /**
   * Logical OR.
   */
  OR("or"),

  PASS("pass"),
  RAISE("raise"),
  RETURN("return"),
  TRY("try"),
  WHILE("while"),
  WITH("with"),
  YIELD("yield");

  private final String value;

  PythonKeyword(String value) {
    this.value = value;
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

  public static String[] keywordValues() {
    PythonKeyword[] keywordsEnum = PythonKeyword.values();
    String[] keywords = new String[keywordsEnum.length];
    for (int i = 0; i < keywords.length; i++) {
      keywords[i] = keywordsEnum[i].getValue();
    }
    return keywords;
  }

}
