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

public class AssertStatementTest extends RuleTest {

  @Before
  public void init() {
    setRootRule(PythonGrammar.ASSERT_STMT);
  }

  @Test
  public void ok() {
    assertThat(p).matches("assert test");
    assertThat(p).matches("assert test , test");
  }

  @Test
  public void realLife() {
    assertThat(p).matches("assert id > 0");
    assertThat(p).matches("assert id > 0, 'id should be positive'");
  }

}
