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
package org.sonar.plugins.python.flake8;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;

public class Flake8ReportReader {

  private static final Logger LOG = Loggers.get(Flake8ReportReader.class);
  private static final Pattern DEFAULT_PATTERN = Pattern.compile("(.+):(\\d+):(\\d+): (\\S+) (.*)");
  private static final Pattern PYLINT_PATTERN = Pattern.compile("(.+):(\\d+): \\[(.*)\\] (.*)");

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

  private static Issue parseLine(String line) {

    if (line.length() > 0) {
      if (!isDetail(line)) {
        Matcher m = DEFAULT_PATTERN.matcher(line);
        if (m.matches()) {
          String filePath = m.group(1);
          int lineNumber = Integer.parseInt(m.group(2));
          int columnNumber = Integer.parseInt(m.group(3));
          String ruleKey = m.group(4);
          String message = m.group(5);
          return new Issue(filePath, ruleKey, message, lineNumber, columnNumber);
        }
        m = PYLINT_PATTERN.matcher(line);
        if (m.matches()) {
          String filePath = m.group(1);
          int lineNumber = Integer.parseInt(m.group(2));
          String ruleKey = m.group(3);
          String message = m.group(4);
          return new Issue(filePath, ruleKey, message, lineNumber, null);
        }
        LOG.debug("Cannot parse the line: {}", line);
      } else {
        LOG.debug("Classifying as detail and ignoring line '{}'", line);
      }
    }
    return null;
  }

  private static boolean isDetail(String line) {
    char first = line.charAt(0);
    return first == ' ' || first == '\t' || first == '\n';
  }

  public static class Issue {

    String filePath;

    String ruleKey;

    String message;

    Integer lineNumber;

    Integer columnNumber;

    public Issue(String filePath, String ruleKey, String message, Integer lineNumber, @Nullable Integer columnNumber) {
      this.filePath = filePath;
      this.ruleKey = ruleKey;
      this.message = message;
      this.lineNumber = lineNumber;
      this.columnNumber = columnNumber;
    }
  }
}
