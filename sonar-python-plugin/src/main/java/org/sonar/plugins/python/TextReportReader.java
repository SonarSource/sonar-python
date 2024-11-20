/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.api.batch.fs.FileSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Common implementation to parse Flake8 and Pylint reports
 */
public class TextReportReader {

  private static final Pattern DEFAULT_PATTERN = Pattern.compile("(.+):(\\d+):(\\d+): (\\S+[^:]):? (.*)");
  private static final Pattern LEGACY_PATTERN = Pattern.compile("(.+):(\\d+): \\[(.*)\\] (.*)");
  private static final Logger LOG = LoggerFactory.getLogger(TextReportReader.class);
  public static final int COLUMN_ZERO_BASED = 0;
  public static final int COLUMN_ONE_BASED = 1;

  private final int reportOffset;

  public TextReportReader(int columnStartIndex) {
    this.reportOffset = columnStartIndex;
  }

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

  private Issue parseLine(String line) {
    if (line.length() > 0) {
      Matcher m = TextReportReader.DEFAULT_PATTERN.matcher(line);
      if (m.matches()) {
        return extractDefaultStyleIssue(m);
      }
      m = TextReportReader.LEGACY_PATTERN.matcher(line);
      if (m.matches()) {
        return extractLegacyStyleIssue(m);
      }
      LOG.debug("Cannot parse the line: {}", line);
    }
    return null;
  }

  private Issue extractDefaultStyleIssue(Matcher m) {
    String filePath = m.group(1);
    int lineNumber = Integer.parseInt(m.group(2));
    int columnNumber = Integer.parseInt(m.group(3));
    // Flake8 column numbering starts at 1
    columnNumber -= this.reportOffset;
    String ruleKey = m.group(4);
    String message = m.group(5);
    return new Issue(filePath, ruleKey, message, lineNumber, columnNumber);
  }

  private static Issue extractLegacyStyleIssue(Matcher m) {
    String filePath = m.group(1);
    int lineNumber = Integer.parseInt(m.group(2));
    String ruleKey = m.group(3);
    int keyLastIndex = ruleKey.indexOf("(");
    if (keyLastIndex > 0) {
      ruleKey = ruleKey.substring(0, keyLastIndex);
    }
    String message = m.group(4);
    return new Issue(filePath, ruleKey, message, lineNumber, null);
  }

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
