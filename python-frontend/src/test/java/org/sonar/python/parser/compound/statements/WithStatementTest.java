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

import org.junit.Before;
import org.junit.jupiter.api.Test;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

public class WithStatementTest extends RuleTest {

  @Before
  public void init() {
    setRootRule(PythonGrammar.WITH_STMT);
  }


  @Test
  public void realLife() {
    assertThat(p).matches(PythonTestUtils.appendNewLine("with A() as a : pass"))
      .matches(PythonTestUtils.appendNewLine("with A() as a, B() as b : pass"));
  }

  @Test
  public void parenthesized_context_managers() {
    assertThat(p).matches(
      "with (open(\"a_really_long_foo\") as foo,\n" +
      "      open(\"a_really_long_bar\") as bar):\n" +
      "        pass");
  }
}
