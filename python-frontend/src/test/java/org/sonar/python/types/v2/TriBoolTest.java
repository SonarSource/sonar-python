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
package org.sonar.python.types.v2;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class TriBoolTest {

  @Test
  void andTest() {
    assertThat(TriBool.TRUE.and(TriBool.TRUE)).isEqualTo(TriBool.TRUE);
    assertThat(TriBool.TRUE.and(TriBool.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.TRUE.and(TriBool.FALSE)).isEqualTo(TriBool.FALSE);
    assertThat(TriBool.FALSE.and(TriBool.TRUE)).isEqualTo(TriBool.FALSE);
    assertThat(TriBool.FALSE.and(TriBool.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.FALSE.and(TriBool.FALSE)).isEqualTo(TriBool.FALSE);
    assertThat(TriBool.UNKNOWN.and(TriBool.TRUE)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.UNKNOWN.and(TriBool.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.UNKNOWN.and(TriBool.FALSE)).isEqualTo(TriBool.UNKNOWN);
  }
}
