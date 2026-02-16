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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class TestFileTelemetryTest {

  @Test
  void emptyTelemetry() {
    var empty = TestFileTelemetry.empty();
    assertThat(empty.totalMainFiles()).isZero();
    assertThat(empty.importBasedMisclassifiedTestFiles()).isZero();
    assertThat(empty.totalLines()).isZero();
    assertThat(empty.totalMainLines()).isZero();
    assertThat(empty.testLines()).isZero();
    assertThat(empty.importBasedMisclassifiedTestLines()).isZero();
    assertThat(empty.pathBasedMisclassifiedTestFiles()).isZero();
    assertThat(empty.pathBasedMisclassifiedTestLines()).isZero();
    assertThat(empty.filesInImportBasedOnly()).isZero();
    assertThat(empty.filesInPathBasedOnly()).isZero();
    assertThat(empty.linesInImportBasedOnly()).isZero();
    assertThat(empty.linesInPathBasedOnly()).isZero();
  }

  @Test
  void addTelemetry() {
    var telemetry1 = new TestFileTelemetry(10, 3, 800, 500, 300, 150, 4, 200, 1, 2, 50, 100);
    var telemetry2 = new TestFileTelemetry(5, 2, 350, 200, 150, 80, 3, 120, 0, 1, 0, 40);

    var combined = telemetry1.add(telemetry2);
    assertThat(combined.totalMainFiles()).isEqualTo(15);
    assertThat(combined.importBasedMisclassifiedTestFiles()).isEqualTo(5);
    assertThat(combined.totalLines()).isEqualTo(1150);
    assertThat(combined.totalMainLines()).isEqualTo(700);
    assertThat(combined.testLines()).isEqualTo(450);
    assertThat(combined.importBasedMisclassifiedTestLines()).isEqualTo(230);
    assertThat(combined.pathBasedMisclassifiedTestFiles()).isEqualTo(7);
    assertThat(combined.pathBasedMisclassifiedTestLines()).isEqualTo(320);
    assertThat(combined.filesInImportBasedOnly()).isEqualTo(1);
    assertThat(combined.filesInPathBasedOnly()).isEqualTo(3);
    assertThat(combined.linesInImportBasedOnly()).isEqualTo(50);
    assertThat(combined.linesInPathBasedOnly()).isEqualTo(140);
  }

  @Test
  void addToEmpty() {
    var empty = TestFileTelemetry.empty();
    var telemetry = new TestFileTelemetry(7, 1, 500, 300, 200, 50, 2, 80, 0, 1, 0, 30);

    var combined = empty.add(telemetry);
    assertThat(combined.totalMainFiles()).isEqualTo(7);
    assertThat(combined.importBasedMisclassifiedTestFiles()).isEqualTo(1);
    assertThat(combined.totalLines()).isEqualTo(500);
    assertThat(combined.totalMainLines()).isEqualTo(300);
    assertThat(combined.testLines()).isEqualTo(200);
    assertThat(combined.importBasedMisclassifiedTestLines()).isEqualTo(50);
    assertThat(combined.pathBasedMisclassifiedTestFiles()).isEqualTo(2);
    assertThat(combined.pathBasedMisclassifiedTestLines()).isEqualTo(80);
    assertThat(combined.filesInImportBasedOnly()).isZero();
    assertThat(combined.filesInPathBasedOnly()).isEqualTo(1);
    assertThat(combined.linesInImportBasedOnly()).isZero();
    assertThat(combined.linesInPathBasedOnly()).isEqualTo(30);
  }
}

