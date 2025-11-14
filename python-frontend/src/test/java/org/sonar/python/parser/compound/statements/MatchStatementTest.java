/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.parser.compound.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

class MatchStatementTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.MATCH_STMT);
  }

  @Test
  void basic_match_statement() {
    assertThat(p).matches(
      """
      match command:
          case "quit":
              ...
          case 42:
              ...
      """);
  }

  @Test
  void match_with_guards() {
    assertThat(p).matches(
      """
      match command:
          case "quit" if True:
              ...
          case "foo" if x:=cond:
              ...
      """);
  }

  @Test
  void subject_expr() {
    setRootRule(PythonGrammar.SUBJECT_EXPR);
    assertThat(p).matches("x := command")
      .matches("x, y")
      .matches("*x")
      .matches("*x, *y")
      .matches("*x, y := command")
      .matches("[x, y := command]")
    ;
  }
}
