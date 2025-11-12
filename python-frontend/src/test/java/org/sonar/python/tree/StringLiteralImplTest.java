/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.tree;

import org.junit.jupiter.api.Test;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;

class StringLiteralImplTest {

  @Test
  void isTemplate() {
    assertThat(stringLiteral("''").isTemplate()).isFalse();
    assertThat(stringLiteral("'abc'").isTemplate()).isFalse();
    assertThat(stringLiteral("t'abc'").isTemplate()).isTrue();
    assertThat(stringLiteral("T'abc'").isTemplate()).isTrue();
    assertThat(stringLiteral("rt'abc'").isTemplate()).isTrue();
    assertThat(stringLiteral("tr'abc'").isTemplate()).isTrue();
  }

  private StringLiteralImpl stringLiteral(String code) {
    return ((StringLiteralImpl) PythonTestUtils.lastExpression(code));
  }
}
