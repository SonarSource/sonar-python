/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.plugins.python.telemetry.collectors;

/**
 * Telemetry data for tracking test file misclassification.
 *
 * @param totalMainFiles Total number of files classified as MAIN
 * @param misclassifiedTestFiles Number of MAIN files that appear to be test files (import unittest/pytest or follow pytest patterns)
 * @param totalLines Total number of lines across all files (MAIN + TEST)
 * @param totalMainLines Total number of lines across all MAIN files
 * @param testLines Number of lines across all TEST files (as classified by the scanner engine)
 * @param misclassifiedTestLines Number of lines across misclassified test files (subset of totalMainLines)
 */
public record TestFileTelemetry(
  long totalMainFiles,
  long misclassifiedTestFiles,
  long totalLines,
  long totalMainLines,
  long testLines,
  long misclassifiedTestLines) {

  public static TestFileTelemetry empty() {
    return new TestFileTelemetry(0, 0, 0, 0, 0, 0);
  }

  public TestFileTelemetry add(TestFileTelemetry other) {
    return new TestFileTelemetry(
      this.totalMainFiles + other.totalMainFiles,
      this.misclassifiedTestFiles + other.misclassifiedTestFiles,
      this.totalLines + other.totalLines,
      this.totalMainLines + other.totalMainLines,
      this.testLines + other.testLines,
      this.misclassifiedTestLines + other.misclassifiedTestLines
    );
  }
}

