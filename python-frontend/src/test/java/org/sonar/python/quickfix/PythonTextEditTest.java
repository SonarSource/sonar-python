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
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.LocationInFile;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonTextEditTest {

  @Test
  public void test(){
    String message = "This is a replacement text";
    LocationInFile loc1 = new LocationInFile(null, 1,7,10,10);
    LocationInFile loc2 = new LocationInFile(null, 1,7,1,7);

    IssueLocation issueLocation1 = IssueLocation.preciseLocation(loc1, "message");
    PythonTextEdit textEdit = PythonTextEdit.insertAtPosition(issueLocation1, message);

    IssueLocation correctLocation = IssueLocation.preciseLocation(loc2, message);

    assertThat(textEdit.replacementText()).isEqualTo(message);
    assertThat(textEdit.issueLocation.startLine()).isEqualTo(correctLocation.startLine());
    assertThat(textEdit.issueLocation.startLineOffset()).isEqualTo(correctLocation.startLineOffset());
    assertThat(textEdit.issueLocation.endLine()).isEqualTo(correctLocation.endLine());
    assertThat(textEdit.issueLocation.endLineOffset()).isEqualTo(correctLocation.endLineOffset());
  }
}
