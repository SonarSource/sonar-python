/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.frontend;

import com.intellij.psi.PsiElement;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonTokenLocationTest {
  private PythonParser parser = new PythonParser();

  @Test
  public void test_multiline() {
    PythonTokenLocation tokenLocation = new PythonTokenLocation(getTokens("'''first line\nsecond'''")[0]);
    assertOffsets(tokenLocation, 1, 0, 2, 9);
  }

  @Test
  public void test_newline_token() {
    PythonTokenLocation tokenLocation = new PythonTokenLocation(getTokens("foo\n")[1]);
    assertOffsets(tokenLocation, 1, 3, 2, 0);
  }

  @Test
  public void test_one_line() {
    PsiElement[] tokens = getTokens("'''first line'''");
    PythonTokenLocation tokenLocation = new PythonTokenLocation(tokens[0]);
    assertOffsets(tokenLocation, 1, 0, 1, 16);

    tokenLocation = new PythonTokenLocation(getTokens("foo")[0]);
    assertOffsets(tokenLocation, 1, 0, 1, 3);
  }

  @Test
  public void test_comment() {
    PythonTokenLocation commentLocation = new PythonTokenLocation(getTokens("#comment\n")[0]);
    assertOffsets(commentLocation, 1, 0, 1, 8);
  }

  private static void assertOffsets(PythonTokenLocation tokenLocation, int startLine, int startLineOffset, int endLine, int endLineOffset) {
    assertThat(tokenLocation.startLine()).as("start line").isEqualTo(startLine);
    assertThat(tokenLocation.startLineOffset()).as("start line offset").isEqualTo(startLineOffset);
    assertThat(tokenLocation.endLine()).as("end line").isEqualTo(endLine);
    assertThat(tokenLocation.endLineOffset()).as("end line offset").isEqualTo(endLineOffset);
  }

  private PsiElement[] getTokens(String toLex) {
    return parser.parse(toLex).getChildren();
  }
}
