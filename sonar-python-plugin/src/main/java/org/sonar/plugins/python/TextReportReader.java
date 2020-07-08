/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import javax.annotation.Nullable;
import org.sonar.api.batch.fs.FileSystem;

public abstract class TextReportReader {

  public List<Issue> parse(File report, FileSystem fileSystem) throws IOException {
    List<Issue> issues = new ArrayList<>();
    try (Scanner scanner = new Scanner(report.toPath(), fileSystem.encoding().name())) {
      while (scanner.hasNextLine()) {
        Issue issue = parseLine(scanner.nextLine());
        if (issue != null) {
          issues.add(issue);
        }
      }
    }
    return issues;
  }

  protected abstract Issue parseLine(String line);

  public static class Issue {

    public final String filePath;

    public final String ruleKey;

    public final String message;

    public final Integer lineNumber;

    public final Integer columnNumber;

    public Issue(String filePath, String ruleKey, String message, Integer lineNumber, @Nullable Integer columnNumber) {
      this.filePath = filePath;
      this.ruleKey = ruleKey;
      this.message = message;
      this.lineNumber = lineNumber;
      this.columnNumber = columnNumber;
    }
  }
}
