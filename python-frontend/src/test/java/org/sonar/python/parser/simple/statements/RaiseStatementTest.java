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
package org.sonar.python.parser.simple.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

class RaiseStatementTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.RAISE_STMT);
  }

  @Test
  void ok() {
    assertThat(p).matches("raise");
    assertThat(p).matches("raise test");

    assertThat(p).matches("raise test, test");
    assertThat(p).matches("raise test, test, test");

    assertThat(p).matches("raise test from test");
  }

  @Test
  void realLife() {
    assertThat(p).matches("raise");
    assertThat(p).matches("raise exc_info[1], None, exc_info[2]");
  }

}
