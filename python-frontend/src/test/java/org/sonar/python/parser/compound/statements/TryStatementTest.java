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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

public class TryStatementTest extends RuleTest {

  @BeforeEach
  public void init() {
    setRootRule(PythonGrammar.TRY_STMT);
  }

  @Test
  public void ok() {

    assertThat(p).matches("try : pass\nexcept e : pass")
      .matches("try : pass\nexcept e : pass\nexcept f : pass")
      .matches("try : pass\nexcept e : pass\nelse : pass")
      .matches("try : pass\nexcept e : pass\nfinally : pass")
      .matches("try : pass\nexcept e : pass\nelse : pass\nfinally : pass")
      .matches("try : pass\nfinally : pass")
      .matches("try : pass\nexcept* e : pass");
  }

}
