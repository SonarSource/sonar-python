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
package org.sonar.plugins.python.api.quickfix;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonQuickFixTest {

  @Test
  void test_newQuickFix_builder() {
    PythonTextEdit textEdit = new PythonTextEdit("This is a replacement text", 1, 2, 3, 4);

    PythonQuickFix quickFix = PythonQuickFix.newQuickFix("New quickfix").addTextEdit(textEdit).build();

    assertThat(quickFix.getTextEdits()).containsExactly(textEdit);
    assertThat(quickFix.getDescription()).isEqualTo("New quickfix");
  }

  @Test
  void test_newQuickFix() {
    PythonTextEdit textEdit = new PythonTextEdit("This is a replacement text", 1, 2, 3, 4);

    PythonQuickFix quickFix = PythonQuickFix.newQuickFix("New quickfix", textEdit);

    assertThat(quickFix.getTextEdits()).containsExactly(textEdit);
  }
}
