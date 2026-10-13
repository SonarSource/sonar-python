/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

class CasePatternsTest extends RuleTest {

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
      .matches("3")
      .matches("-3")
      .matches("+3")
      .matches("+ 3.14")
      .matches("+0xFF")
      .matches("+42j")
      .notMatches("++3")
      .notMatches("+-3")
      .notMatches("-+3")
      .notMatches("+value")
      .notMatches("+True")
      .notMatches("+(3)");
  }

  @Test
  void complex_numbers() {
    assertThat(p)
      .matches("3 + 4j")
      .matches("+3 + 4j")
      .matches("+3 - 4j")
      .matches("-3 + 4j")
      .notMatches("+3 + +4j")
      .notMatches("+3 - -4j")
      .matches("42j");
  }

  @Test
  void signed_numbers_in_nested_patterns() {
    assertThat(p)
      .matches("[+3, +4.5]")
      .matches("{+3: value, +4 - 5j: other}")
      .matches("+3 | -3")
      .matches("+3 as value");
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
