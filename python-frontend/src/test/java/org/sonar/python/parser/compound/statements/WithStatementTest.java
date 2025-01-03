/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.parser.compound.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

class WithStatementTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.WITH_STMT);
  }


  @Test
  void realLife() {
    assertThat(p).matches(PythonTestUtils.appendNewLine("with A() as a : pass"))
      .matches(PythonTestUtils.appendNewLine("with A() as a, B() as b : pass"));
  }

  @Test
  void parenthesized_context_managers() {
    assertThat(p).matches(
      "with (open(\"a_really_long_foo\") as foo,\n" +
      "      open(\"a_really_long_bar\") as bar):\n" +
      "        pass");
  }
}
