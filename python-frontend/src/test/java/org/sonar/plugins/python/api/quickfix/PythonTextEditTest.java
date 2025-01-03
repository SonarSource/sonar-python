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
package org.sonar.plugins.python.api.quickfix;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;

import static org.assertj.core.api.Assertions.assertThat;

class PythonTextEditTest {

  @Test
  void equals() {
    PythonTextEdit edit = new PythonTextEdit("", 0, 0, 1, 1);
    assertThat(edit.equals(edit)).isTrue();
    assertThat(edit.equals(null)).isFalse();
    assertThat(edit.equals(new Object())).isFalse();

    assertThat(edit.equals(new PythonTextEdit("", 0, 0, 1, 1))).isTrue();
    assertThat(edit.equals(new PythonTextEdit("",1, 0, 1, 1))).isFalse();
    assertThat(edit.equals(new PythonTextEdit("",0, 1, 1, 1))).isFalse();
    assertThat(edit.equals(new PythonTextEdit("",0, 0, 0, 1))).isFalse();
    assertThat(edit.equals(new PythonTextEdit("",0, 0, 1, 0))).isFalse();
    assertThat(edit.equals(new PythonTextEdit("a", 0, 0, 1, 1))).isFalse();
  }

  @Test
  void test_hashCode() {
    PythonTextEdit edit = new PythonTextEdit("", 0, 0, 1, 1);
    assertThat(edit)
      .hasSameHashCodeAs(edit)
      .hasSameHashCodeAs(new PythonTextEdit("", 0, 0, 1, 1))
      .doesNotHaveSameHashCodeAs(new Object())
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("",1, 0, 1, 1))
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("",0, 1, 1, 1))
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("",0, 0, 0, 1))
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("",0, 0, 1, 0))
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("a", 0, 0, 1, 1))
    ;
  }
}
