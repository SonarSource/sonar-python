/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.parser.simple.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

public class ExpressionStatementTest extends RuleTest {

  @BeforeEach
  public void init() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
  }

  @Test
  public void realLife() {
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
