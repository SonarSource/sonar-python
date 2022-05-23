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

import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.LocationInFile;

public class PythonTextEdit {

  public final IssueLocation issueLocation;

  public PythonTextEdit(LocationInFile location, String addition) {
    this.issueLocation = IssueLocation.preciseLocation(location, addition);
  }

  public static PythonTextEdit insertAtPosition(IssueLocation issueLocation, String addition) {
    LocationInFile location = atBeginningOfIssue(issueLocation);
    return new PythonTextEdit(location, addition);
  }

  private static LocationInFile atBeginningOfIssue(IssueLocation issue) {
    return new LocationInFile(issue.fileId(), issue.startLine(), issue.startLineOffset(), issue.startLine(), issue.startLineOffset());
  }

  public String message() {
    return issueLocation.message();
  }
}
