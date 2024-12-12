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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.sensor.SensorContext;

public class SensorTelemetryStorage {
  private static final Logger LOG = LoggerFactory.getLogger(SensorTelemetryStorage.class);

  public static final MetricKey NOTEBOOK_PRESENT_KEY = new MetricKey("python.notebook.present");
  public static final MetricKey NOTEBOOK_TOTAL_KEY = new MetricKey("python.notebook.total");
  public static final MetricKey NOTEBOOK_RECOGNITION_ERROR_KEY = new MetricKey("python.notebook.recognition_error");
  public static final MetricKey NOTEBOOK_EXCEPTION_KEY = new MetricKey("python.notebook.exceptions");

  private final Map<String, String> data;

  public SensorTelemetryStorage() {
    data = new HashMap<>();
  }

  public Map<String, String> data() {
    return Collections.unmodifiableMap(data);
  }

  public void send(SensorContext sensorContext) {
    data.forEach(sensorContext::addTelemetryProperty);
    data.forEach((k, v) -> LOG.info("Metrics property: {}={}", k, v));
  }

  public void updateMetric(MetricKey key, String value) {
    data.put(key.key(), value);
  }

  static class MetricKey {
    private final String key;

    private MetricKey(String key) {
      this.key = key;
    }

    public String key() {
      return key;
    }
  }
}
