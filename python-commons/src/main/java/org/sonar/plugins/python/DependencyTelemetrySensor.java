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
package org.sonar.plugins.python;

import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.plugins.python.dependency.DependencyTelemetry;
import org.sonar.plugins.python.telemetry.SensorTelemetryStorage;

public class DependencyTelemetrySensor implements Sensor {
  private final SensorTelemetryStorage sensorTelemetryStorage = new SensorTelemetryStorage();

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .name("Python Dependency Sensor");
  }

  @Override
  public void execute(SensorContext context) {
    new DependencyTelemetry(sensorTelemetryStorage, context.fileSystem()).process();
    sensorTelemetryStorage.send(context);
  }
}
