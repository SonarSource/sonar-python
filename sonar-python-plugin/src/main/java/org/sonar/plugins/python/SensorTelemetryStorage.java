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

import java.util.EnumMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.utils.Version;

public class SensorTelemetryStorage {
  private static final Logger LOG = LoggerFactory.getLogger(SensorTelemetryStorage.class);

  private final Map<TelemetryMetricKey, String> data = new EnumMap<>(TelemetryMetricKey.class);

  public void send(SensorContext sensorContext) {
    // This try/catch block should be useless, as in the worst case it should be a no-op depending on the SensorContext implementation
    // It exists to be extra sure for the LTA
    try {
      var apiVersion = sensorContext.runtime().getApiVersion();
      if (apiVersion.isGreaterThanOrEqual(Version.create(10, 9))) {
        data.forEach((k, v) -> {
          LOG.info("Collected metric: {}={}", k, v);
          sensorContext.addTelemetryProperty(k.key(), v);
        });

      } else {
        LOG.info("Skipping sending metrics because the plugin API version is {}", apiVersion);
      }
    } catch (Exception e) {
      LOG.error("Failed to send metrics", e);
    }
  }

  public void updateMetric(TelemetryMetricKey key, String value) {
    data.put(key, value);
  }

  public void updateMetric(TelemetryMetricKey key, int value) {
    data.put(key, String.valueOf(value));
  }

  public void updateMetric(TelemetryMetricKey key, boolean value) {
    data.put(key, boolToString(value));
  }

  private static String boolToString(boolean value) {
    return value ? "1" : "0";
  }

}
