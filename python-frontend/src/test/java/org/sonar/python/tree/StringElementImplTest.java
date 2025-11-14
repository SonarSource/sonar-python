/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;

class StringElementImplTest {

  @Test
  void isInterpolated() {
    assertThat(stringElement("'abc'").isInterpolated()).isFalse();
    assertThat(stringElement("f'abc'").isInterpolated()).isTrue();
    assertThat(stringElement("F'abc'").isInterpolated()).isTrue();
    assertThat(stringElement("r'abc'").isInterpolated()).isFalse();
    assertThat(stringElement("rf'abc'").isInterpolated()).isTrue();
  }

  @Test
  void isTemplate() {
    assertThat(stringElement("'abc'").isTemplate()).isFalse();
    assertThat(stringElement("t'abc'").isTemplate()).isTrue();
    assertThat(stringElement("T'abc'").isTemplate()).isTrue();
    assertThat(stringElement("rt'abc'").isTemplate()).isTrue();
    assertThat(stringElement("tr'abc'").isTemplate()).isTrue();
  }

  private StringElement stringElement(String code) {
    return ((StringLiteral) PythonTestUtils.lastExpression(code)).stringElements().get(0);
  }
}
