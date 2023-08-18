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
package org.sonar.python.parser.compound.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

public class CasePatternsTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.PATTERN);
  }

  @Test
  void strings() {
    assertThat(p)
      .matches("\"foo\"")
      .matches("\"foo\" \"bar\"");
  }

  @Test
  void none() {
    assertThat(p).matches("None");
  }

  @Test
  void booleans() {
    setRootRule(PythonGrammar.LITERAL_PATTERN);
    assertThat(p)
      .matches("True")
      .matches("False")
      .notMatches("Other");
  }

  @Test
  void signed_numbers() {
    assertThat(p)
      .matches("-3")
      .notMatches("+3");
  }

  @Test
  void complex_numbers() {
    assertThat(p)
      .matches("3 + 4j")
      .matches("42j");
  }

  @Test
  void as_patterns() {
    assertThat(p)
      .matches("'foo' as x")
      .matches("x as y")
      .notMatches("'foo' as x as y");
  }

  @Test
  void capture() {
    assertThat(p)
      .matches("x");
  }
}
