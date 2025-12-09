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
    assertThat(empty.misclassifiedTestFiles()).isZero();
  }

  @Test
  void addTelemetry() {
    var telemetry1 = new TestFileTelemetry(10, 3);
    var telemetry2 = new TestFileTelemetry(5, 2);

    var combined = telemetry1.add(telemetry2);
    assertThat(combined.totalMainFiles()).isEqualTo(15);
    assertThat(combined.misclassifiedTestFiles()).isEqualTo(5);
  }

  @Test
  void addToEmpty() {
    var empty = TestFileTelemetry.empty();
    var telemetry = new TestFileTelemetry(7, 1);

    var combined = empty.add(telemetry);
    assertThat(combined.totalMainFiles()).isEqualTo(7);
    assertThat(combined.misclassifiedTestFiles()).isEqualTo(1);
  }
}

