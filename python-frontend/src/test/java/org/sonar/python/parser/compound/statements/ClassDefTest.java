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
package org.sonar.python.parser.compound.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

class ClassDefTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.CLASSDEF);
  }

  @Test
  void realLife() {
    assertThat(p).matches(PythonTestUtils.appendNewLine("class Foo: pass"));
    assertThat(p).matches(PythonTestUtils.appendNewLine("class Foo(argument): pass"));
    assertThat(p).matches(PythonTestUtils.appendNewLine("class Foo(argument=value): pass"));
    assertThat(p).matches(PythonTestUtils.appendNewLine("class Foo: x: int"));
    assertThat(p).matches(PythonTestUtils.appendNewLine("class Foo: x: int = 3"));
  }

}
