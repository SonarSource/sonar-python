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
import org.sonar.python.parser.RuleTest;

import static org.sonar.sslr.tests.Assertions.assertThat;

public class TryStatementTest extends RuleTest {

  @Before
  public void init() {
    setRootRule(PythonGrammar.TRY_STMT);
  }

  @Test
  public void ok() {
    p.getGrammar().rule(PythonGrammar.SUITE).mock();
    p.getGrammar().rule(PythonGrammar.TEST).mock();
    p.getGrammar().rule(PythonGrammar.EXCEPT_CLAUSE).mock();

    assertThat(p).matches("try : SUITE EXCEPT_CLAUSE : SUITE");
    assertThat(p).matches("try : SUITE EXCEPT_CLAUSE : SUITE EXCEPT_CLAUSE : SUITE");
    assertThat(p).matches("try : SUITE EXCEPT_CLAUSE : SUITE else : SUITE");
    assertThat(p).matches("try : SUITE EXCEPT_CLAUSE : SUITE finally : SUITE");
    assertThat(p).matches("try : SUITE EXCEPT_CLAUSE : SUITE else : SUITE finally : SUITE");
    assertThat(p).matches("try : SUITE finally : SUITE");
  }

}
