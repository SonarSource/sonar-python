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

class SimpleStatementTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.SIMPLE_STMT);
  }

  @Test
  void ok() {
    assertThat(p).matches("PRINT_STMT");
    assertThat(p).matches("EXEC_STMT");
    assertThat(p).matches("EXPRESSION_STMT");
    assertThat(p).matches("ASSERT_STMT");
    assertThat(p).matches("PASS_STMT");
    assertThat(p).matches("DEL_STMT");
    assertThat(p).matches("RETURN_STMT");
    assertThat(p).matches("YIELD_STMT");
    assertThat(p).matches("RAISE_STMT");
    assertThat(p).matches("BREAK_STMT");
    assertThat(p).matches("CONTINUE_STMT");
    assertThat(p).matches("IMPORT_STMT");
    assertThat(p).matches("GLOBAL_STMT");
    assertThat(p).matches("NONLOCAL_STMT");
  }

  @Test
  void realLife() {
    assertThat(p).matches("print 'Hello world'");
    assertThat(p).matches("print = 12");
    assertThat(p).matches("exec 'print 1'");
    assertThat(p).matches("i += 1");
    assertThat(p).matches("print('something', file=out_file)");
  }

}
