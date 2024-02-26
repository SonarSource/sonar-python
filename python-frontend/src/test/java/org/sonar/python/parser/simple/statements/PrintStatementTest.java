/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.parser.simple.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

class PrintStatementTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.PRINT_STMT);
  }

  @Test
  void ok() {
    assertThat(p).matches("print")

      .matches("print >> test")
      .matches("print >> test, test")
      .matches("print >> test, test,")

      .matches("print test")
      .matches("print test,")
      .matches("print test,test")
      .matches("print test,test,")

      .notMatches("print >>")

      .matches("print 1")
      .matches("print 1,")
      .matches("print >> 1")
      .notMatches("print('')")
      .notMatches("print = 12")
    ;
  }

}
