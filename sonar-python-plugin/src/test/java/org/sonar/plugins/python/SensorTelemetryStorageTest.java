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
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.SonarEdition;
import org.sonar.api.SonarQubeSide;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.api.utils.Version;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

class SensorTelemetryStorageTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @Test
  void no_send_on_incompatible_version() {
    var sensorContext = sensorContext(Version.create(10, 8));
    var storage = new SensorTelemetryStorage();
    storage.updateMetric(TelemetryMetricKey.NOTEBOOK_PRESENT_KEY, "1");
    storage.send(sensorContext);

    verify(sensorContext, never()).addTelemetryProperty(any(), any());
    assertThat(logTester.logs()).contains("Skipping sending metrics because the plugin API version is 10.8");
  }

  @Test
  void send_after_10_9() {
    var sensorContext = sensorContext(Version.create(10, 9));
    var storage = new SensorTelemetryStorage();
    storage.updateMetric(TelemetryMetricKey.NOTEBOOK_PRESENT_KEY, "1");
    storage.send(sensorContext);

    verify(sensorContext).addTelemetryProperty(TelemetryMetricKey.NOTEBOOK_PRESENT_KEY.key(), "1");
  }

  @Test
  void no_crash_on_exception() {
    var sensorContext = sensorContext(Version.create(10, 9));
    doThrow(new RuntimeException("Some exception")).when(sensorContext).addTelemetryProperty(any(), any());
    var storage = new SensorTelemetryStorage();
    storage.updateMetric(TelemetryMetricKey.NOTEBOOK_PRESENT_KEY, "1");
    Assertions.assertDoesNotThrow(() -> storage.send(sensorContext));
    assertThat(logTester.logs()).contains("Failed to send metrics");
  }

  private SensorContext sensorContext(Version version) {
    var sensorContext = spy(SensorContextTester.create(new File("")));
    sensorContext.setRuntime(SonarRuntimeImpl.forSonarQube(version, SonarQubeSide.SERVER, SonarEdition.DEVELOPER));
    return sensorContext;
  }
}
