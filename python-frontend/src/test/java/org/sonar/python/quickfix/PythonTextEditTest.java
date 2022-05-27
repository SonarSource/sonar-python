/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.quickfix;

import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.tree.Token;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

public class PythonTextEditTest {

  @Test
  public void test() {
    String replacementText = "This is a replacement text";
    LocationInFile finalLocation = new LocationInFile("null", 1, 7, 1, 7);

    Token token = Mockito.mock(Token.class);
    when(token.line()).thenReturn(1);
    when(token.column()).thenReturn(7);
    when(token.firstToken()).thenReturn(token);
    when(token.lastToken()).thenReturn(token);

    PythonTextEdit textEdit = PythonTextEdit.insertBefore(token, replacementText);

    assertThat(textEdit.replacementText()).isEqualTo(replacementText);
    assertThat(textEdit.startLine()).isEqualTo(1);
    assertThat(textEdit.startLineOffset()).isEqualTo(7);
    assertThat(textEdit.endLine()).isEqualTo(1);
    assertThat(textEdit.endLineOffset()).isEqualTo(7);
  }
}
