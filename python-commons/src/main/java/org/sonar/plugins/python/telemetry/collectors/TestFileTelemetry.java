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
 * @param importBasedMisclassifiedTestFiles Number of MAIN files that appear to be test files based on import heuristic (import unittest/pytest or follow pytest patterns)
 * @param totalLines Total number of lines across all files (MAIN + TEST)
 * @param totalMainLines Total number of lines across all MAIN files
 * @param testLines Number of lines across all TEST files (as classified by the scanner engine)
 * @param importBasedMisclassifiedTestLines Number of lines across import-based misclassified test files (subset of totalMainLines)
 * @param pathBasedMisclassifiedTestFiles Number of MAIN files that appear to be test files based on path heuristic (contains "test" or "tests" in path)
 * @param pathBasedMisclassifiedTestLines Number of lines across path-based misclassified test files
 * @param filesInImportBasedOnly Number of files detected by import-based heuristic but not by path-based heuristic
 * @param filesInPathBasedOnly Number of files detected by path-based heuristic but not by import-based heuristic
 * @param linesInImportBasedOnly Number of lines in files detected by import-based heuristic but not by path-based heuristic
 * @param linesInPathBasedOnly Number of lines in files detected by path-based heuristic but not by import-based heuristic
 */
public record TestFileTelemetry(
  long totalMainFiles,
  long importBasedMisclassifiedTestFiles,
  long totalLines,
  long totalMainLines,
  long testLines,
  long importBasedMisclassifiedTestLines,
  long pathBasedMisclassifiedTestFiles,
  long pathBasedMisclassifiedTestLines,
  long filesInImportBasedOnly,
  long filesInPathBasedOnly,
  long linesInImportBasedOnly,
  long linesInPathBasedOnly) {

  public static TestFileTelemetry empty() {
    return new TestFileTelemetry(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  }

  public TestFileTelemetry add(TestFileTelemetry other) {
    return new TestFileTelemetry(
      this.totalMainFiles + other.totalMainFiles,
      this.importBasedMisclassifiedTestFiles + other.importBasedMisclassifiedTestFiles,
      this.totalLines + other.totalLines,
      this.totalMainLines + other.totalMainLines,
      this.testLines + other.testLines,
      this.importBasedMisclassifiedTestLines + other.importBasedMisclassifiedTestLines,
      this.pathBasedMisclassifiedTestFiles + other.pathBasedMisclassifiedTestFiles,
      this.pathBasedMisclassifiedTestLines + other.pathBasedMisclassifiedTestLines,
      this.filesInImportBasedOnly + other.filesInImportBasedOnly,
      this.filesInPathBasedOnly + other.filesInPathBasedOnly,
      this.linesInImportBasedOnly + other.linesInImportBasedOnly,
      this.linesInPathBasedOnly + other.linesInPathBasedOnly
    );
  }
}

