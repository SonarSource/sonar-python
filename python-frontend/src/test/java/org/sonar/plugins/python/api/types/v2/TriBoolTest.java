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
package org.sonar.plugins.python.api.types.v2;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.TriBool;

import static org.assertj.core.api.Assertions.assertThat;

class TriBoolTest {

  @Test
  void conservativeAndTest() {
    assertThat(TriBool.TRUE.conservativeAnd(TriBool.TRUE)).isEqualTo(TriBool.TRUE);
    assertThat(TriBool.TRUE.conservativeAnd(TriBool.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.TRUE.conservativeAnd(TriBool.FALSE)).isEqualTo(TriBool.FALSE);
    assertThat(TriBool.FALSE.conservativeAnd(TriBool.TRUE)).isEqualTo(TriBool.FALSE);
    assertThat(TriBool.FALSE.conservativeAnd(TriBool.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.FALSE.conservativeAnd(TriBool.FALSE)).isEqualTo(TriBool.FALSE);
    assertThat(TriBool.UNKNOWN.conservativeAnd(TriBool.TRUE)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.UNKNOWN.conservativeAnd(TriBool.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.UNKNOWN.conservativeAnd(TriBool.FALSE)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void orTest() {
    assertThat(TriBool.TRUE.or(TriBool.TRUE)).isEqualTo(TriBool.TRUE);
    assertThat(TriBool.TRUE.or(TriBool.UNKNOWN)).isEqualTo(TriBool.TRUE);
    assertThat(TriBool.TRUE.or(TriBool.FALSE)).isEqualTo(TriBool.TRUE);
    assertThat(TriBool.FALSE.or(TriBool.TRUE)).isEqualTo(TriBool.TRUE);
    assertThat(TriBool.FALSE.or(TriBool.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.FALSE.or(TriBool.FALSE)).isEqualTo(TriBool.FALSE);
    assertThat(TriBool.UNKNOWN.or(TriBool.TRUE)).isEqualTo(TriBool.TRUE);
    assertThat(TriBool.UNKNOWN.or(TriBool.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TriBool.UNKNOWN.or(TriBool.FALSE)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void isTrueTest() {
    assertThat(TriBool.TRUE.isTrue()).isTrue();
    assertThat(TriBool.FALSE.isTrue()).isFalse();
    assertThat(TriBool.UNKNOWN.isTrue()).isFalse();
  }

  @Test
  void isFalseTest() {
    assertThat(TriBool.TRUE.isFalse()).isFalse();
    assertThat(TriBool.FALSE.isFalse()).isTrue();
    assertThat(TriBool.UNKNOWN.isFalse()).isFalse();
  }
}
