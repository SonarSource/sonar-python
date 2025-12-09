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
 * @param misclassifiedTestFiles Number of MAIN files that import unittest or pytest (likely test files)
 */
public record TestFileTelemetry(
  long totalMainFiles,
  long misclassifiedTestFiles) {

  public static TestFileTelemetry empty() {
    return new TestFileTelemetry(0, 0);
  }

  public TestFileTelemetry add(TestFileTelemetry other) {
    return new TestFileTelemetry(
      this.totalMainFiles + other.totalMainFiles,
      this.misclassifiedTestFiles + other.misclassifiedTestFiles
    );
  }
}

