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

class AssertStatementTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.ASSERT_STMT);
  }

  @Test
  void ok() {
    assertThat(p).matches("assert test");
    assertThat(p).matches("assert test , test");
  }

  @Test
  void realLife() {
    assertThat(p).matches("assert id > 0");
    assertThat(p).matches("assert id > 0, 'id should be positive'");
  }

}
