/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.python.parser.python_v3;

import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.sslr.tests.Assertions.assertThat;

public class PythonV3Test extends RuleTest {

  @Test
  public void ellipsis(){
    setRootRule(PythonGrammar.TEST);
    assertThat(p).matches("...");
    assertThat(p).matches("x[...]");
  }

  @Test
  public void function_declaration(){
    setRootRule(PythonGrammar.FUNCDEF);
    assertThat(p).matches("def fun()->'Returns some value': pass");
    assertThat(p).matches("def fun(count:'Number of smth'=1, value:'Value of smth', type:Type): pass");

    setRootRule(PythonGrammar.LAMBDEF);
    assertThat(p).matches("lambda x: x");
    assertThat(p).notMatches("lambda x:Type: x");
  }

}
