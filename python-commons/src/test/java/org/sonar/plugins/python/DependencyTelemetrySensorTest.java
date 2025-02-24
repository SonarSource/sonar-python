/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.nio.file.Paths;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

class DependencyTelemetrySensorTest {
  @Test
  void testDescribe() {
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    new DependencyTelemetrySensor().describe(descriptor);

    assertThat(descriptor.name()).isEqualTo("Python Dependency Sensor");
    assertThat(descriptor.languages()).containsOnly("txt", "toml");
    assertThat(descriptor.type()).isNull();
  }

  @Test
  void testExecute() {
    SensorContextTester context = spy(SensorContextTester.create(Paths.get(".")));
    new DependencyTelemetrySensor().execute(context);
    verify(context).addTelemetryProperty(TelemetryMetricKey.PYTHON_DEPENDENCIES.key(), "");
  }

}
