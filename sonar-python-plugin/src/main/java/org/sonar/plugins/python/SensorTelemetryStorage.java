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

import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.sensor.SensorContext;

public class SensorTelemetryStorage {
  private static final Logger LOG = LoggerFactory.getLogger(SensorTelemetryStorage.class);

  private final Map<String, String> data = new HashMap<>();

  public void send(SensorContext sensorContext) {
    data.forEach((k, v) -> {
      LOG.info("Metrics property: {}={}", k, v);
      sensorContext.addTelemetryProperty(k, v);
    });
  }

  public void updateMetric(MetricKey key, String value) {
    data.put(key.key(), value);
  }

  public enum MetricKey {
    NOTEBOOK_PRESENT_KEY("python.notebook.present"),
    NOTEBOOK_TOTAL_KEY("python.notebook.total"),
    NOTEBOOK_RECOGNITION_ERROR_KEY("python.notebook.recognition_error"),
    NOTEBOOK_EXCEPTION_KEY("python.notebook.exceptions");

    private final String key;

    MetricKey(String key) {
      this.key = key;
    }

    public String key() {
      return key;
    }
  }
}
