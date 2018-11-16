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
package org.sonar.python.parser.compound_statements;

import org.junit.Before;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.PythonTestUtils;
import org.sonar.python.parser.RuleTest;

import static org.sonar.sslr.tests.Assertions.assertThat;

public class ForStatementTest extends RuleTest {

  @Before
  public void init() {
    setRootRule(PythonGrammar.FOR_STMT);
  }

  @Test
  public void ok() {
    p.getGrammar().rule(PythonGrammar.EXPRLIST).mock();
    p.getGrammar().rule(PythonGrammar.TESTLIST).mock();
    p.getGrammar().rule(PythonGrammar.SUITE).mock();

    assertThat(p).matches("for EXPRLIST in TESTLIST : SUITE");
    assertThat(p).matches("for EXPRLIST in TESTLIST : SUITE else : SUITE");
  }

  @Test
  public void realLife() {
    assertThat(p).matches(PythonTestUtils.appendNewLine("for i in [0,2] : pass"));
    assertThat(p).matches(PythonTestUtils.appendNewLine("for x in [0,10] : print(x)"));
  }

}
