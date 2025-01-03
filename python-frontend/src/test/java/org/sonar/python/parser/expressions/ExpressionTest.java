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
package org.sonar.python.parser.expressions;

import com.sonar.sslr.api.AstNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.parser.PythonParserAssert.assertThat;

class ExpressionTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.TEST);
  }

  @Test
  void realLife() {
    assertThat(p).matches("1 + 2 * 3");
    assertThat(p).matches("(1 + 1) * 2");

    assertThat(p).matches("True");
    assertThat(p).matches("False");
    assertThat(p).matches("None");

    assertThat(p).matches("list[1]");
    assertThat(p).matches("list[1:3]");
    assertThat(p).matches("list[:]");

    assertThat(p).matches("[1, 2]");
    assertThat(p).matches("[1, 2,]");

    assertThat(p).matches("{'foo': 1, 'bar': 2, 'baz': 3,}");
    assertThat(p).matches("{'foo': 1, 'bar': 2, 'baz': 3,}");

    assertThat(p).matches("print(something,)");

    assertThat(p).matches("func(value, parameter = value)");

    assertThat(p).matches("lambda x: x**2");

    assertThat(p).matches("[x**2 for x in range(10)]");
    assertThat(p).matches("[x**2 for x in 1, 2, 3]")
      .matches("[x**2 async for x in 1, 2, 3]");
  }

  @Test
  void call_expression() {
    AstNode powerNode = p.parse("f()**g()").getFirstChild(PythonGrammar.POWER);
    assertThat(powerNode).isNotNull();
  }

}
