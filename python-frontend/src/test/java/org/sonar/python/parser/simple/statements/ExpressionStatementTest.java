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
package org.sonar.python.parser.simple.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

class ExpressionStatementTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
  }

  @Test
  void realLife() {
    assertThat(p)
      .matches("i = 10")
      .matches("list[1] = 10")
      .matches("self.balance = initial_balance")
      .matches("var1: int = 5")
      .matches("var2: [int, str]")
      .matches("st: str = 'Hello'")
      .matches("a.b: int = (1, 2)")
      .matches("x: int")
      .matches("self.x: int = x")
      .matches("lst: List[int] = []")
      .matches("lst: List[int] = []")
      .matches("print = 12")
      .notMatches("print 1")
      // EXPRESSION_STMT matches this but that should be match by print statement first.
      .matches("print >> test")
    ;
  }

}
