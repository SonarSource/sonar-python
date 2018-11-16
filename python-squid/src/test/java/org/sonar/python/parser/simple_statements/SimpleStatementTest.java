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
package org.sonar.python.parser.simple_statements;

import org.junit.Before;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.sslr.tests.Assertions.assertThat;

public class SimpleStatementTest extends RuleTest {

  @Before
  public void init() {
    setRootRule(PythonGrammar.SIMPLE_STMT);
  }

  @Test
  public void ok() {
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
  public void realLife() {
    assertThat(p).matches("print 'Hello world'");
    assertThat(p).matches("exec 'print 1'");
    assertThat(p).matches("i += 1");
    assertThat(p).matches("print('something', file=out_file)");
  }

}
