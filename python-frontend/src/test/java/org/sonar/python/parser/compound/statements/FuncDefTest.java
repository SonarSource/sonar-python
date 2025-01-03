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
package org.sonar.python.parser.compound.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

class FuncDefTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.FUNCDEF);
  }

  @Test
  void realLife() {
    assertThat(p).matches(PythonTestUtils.appendNewLine("def func(): pass"));
  }

  @Test
  void trueAsParameter() {
    assertThat(p).matches(PythonTestUtils.appendNewLine("def func(True): pass"));
  }

  @Test
  void trailingComa() {
    assertThat(p).matches(PythonTestUtils.appendNewLine("def func(self, arg1, arg2, arg3, arg4, arg5, arg6, *args, **kwargs,): pass"));
  }

}
