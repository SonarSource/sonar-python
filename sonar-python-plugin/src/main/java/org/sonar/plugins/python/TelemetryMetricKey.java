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

public enum TelemetryMetricKey {
  NOTEBOOK_PRESENT_KEY("python.notebook.present"),
  NOTEBOOK_TOTAL_KEY("python.notebook.total"),
  NOTEBOOK_RECOGNITION_ERROR_KEY("python.notebook.recognition_error"),
  NOTEBOOK_EXCEPTION_KEY("python.notebook.exceptions"),
  PYTHON_VERSION_SET_KEY("python.version.set"),
  PYTHON_VERSION_KEY("python.version");

  private final String key;

  TelemetryMetricKey(String key) {
    this.key = key;
  }

  public String key() {
    return key;
  }
}
