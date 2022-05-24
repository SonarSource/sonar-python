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

import java.util.Arrays;
import org.junit.Test;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.LocationInFile;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonQuickFixTest {

  @Test
  public void test() {
    PythonQuickFix.Builder builder = PythonQuickFix.newQuickFix("New quickfix");
    String message = "This is a replacement text";
    LocationInFile loc1 = new LocationInFile(null, 1, 7, 10, 10);

    IssueLocation issueLocation1 = IssueLocation.preciseLocation(loc1, "message");
    PythonTextEdit textEdit = PythonTextEdit.insertAtPosition(issueLocation1, message);

    PythonQuickFix quickFix = builder.addTextEdit(textEdit).build();

    assertThat(quickFix.getTextEdits()).hasSize(1);
    assertThat(quickFix.getTextEdits().get(0)).isEqualTo(textEdit);
    assertThat(quickFix.getDescription()).isEqualTo("New quickfix");

    LocationInFile loc2 = new LocationInFile(null, 14, 7, 17, 7);
    IssueLocation issueLocation2 = IssueLocation.preciseLocation(loc2, "message");
    PythonTextEdit textEdit2 = PythonTextEdit.insertAtPosition(issueLocation2, message);
    PythonQuickFix quickFix1 = PythonQuickFix.newQuickFix("Second Quickfix")
      .addTextEdits(Arrays.asList(textEdit, textEdit2))
      .build();

    assertThat(quickFix1.getTextEdits()).hasSize(2);
    assertThat(quickFix1.getTextEdits().get(1)).isEqualTo(textEdit2);
    assertThat(quickFix1.getDescription()).isEqualTo("Second Quickfix");
  }

  @Test
  public void multiple_quickfixes() {

  }
}
