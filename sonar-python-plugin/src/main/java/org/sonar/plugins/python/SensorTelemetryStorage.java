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

  public static final String NOTEBOOK_PRESENT_KEY = "python.notebook.present";
  public static final String NOTEBOOK_TOTAL_KEY = "python.notebook.total";
  public static final String NOTEBOOK_PARSE_ERROR_KEY = "python.notebook.parse_error";
  public static final String NOTEBOOK_RECOGNITION_ERROR_KEY = "python.notebook.recognition_error";
  public static final String NOTEBOOK_EXCEPTION_KEY = "python.notebook.exceptions";
  final Map<String, String> data;

  public SensorTelemetryStorage() {
    data = new HashMap<>();
    data.put(NOTEBOOK_PRESENT_KEY, "false");
  }

  public void send(SensorContext sensorContext) {
    data.forEach(sensorContext::addTelemetryProperty);
    data.forEach((k, v) -> LOG.info("Metrics property: {}={}", k, v));
  }
}
