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

public enum TelemetryMetricKey {
  NOTEBOOK_PRESENT_KEY("python.notebook.present"),
  NOTEBOOK_TOTAL_KEY("python.notebook.total"),
  NOTEBOOK_RECOGNITION_ERROR_KEY("python.notebook.recognition_error"),
  NOTEBOOK_EXCEPTION_KEY("python.notebook.exceptions"),
  PYTHON_VERSION_SET_KEY("python.version.set"),
  PYTHON_VERSION_KEY("python.version"),
  PYTHON_DATABRICKS_FOUND("python.notebook.databricks.python"),
  IPYNB_DATABRICKS_FOUND("python.notebook.databricks.ipynb"),
  PYTHON_DEPENDENCIES("python.dependencies"),
  PYTHON_DEPENDENCIES_FORMAT_VERSION("python.dependencies.format_version"),
  NOSONAR_RULE_ID_KEY("python.nosonar.rule_ids"),
  NOSONAR_COMMENTS_KEY("python.nosonar.comments"),
  NOSONAR_NOTEBOOK_RULE_ID_KEY("python.notebook.nosonar.rule_ids"),
  NOSONAR_NOTEBOOK_COMMENTS_KEY("python.notebook.nosonar.comments"),
  ANALYSIS_THREADS_PARAM_KEY("python.analysis.threads.parameter"),
  ANALYSIS_THREADS_KEY("python.analysis.threads.actual"),
  PARALLEL_ANALYSIS_KEY("python.analysis.parallel"),
  ANALYSIS_DURATION_KEY("python.analysis.duration"),
  NOTEBOOKS_ANALYSIS_DURATION_KEY("python.notebooks.analysis.duration"),
  PYTHON_NUMBER_OF_FILES_KEY("python.files.count");

  private final String key;

  TelemetryMetricKey(String key) {
    this.key = key;
  }

  public String key() {
    return key;
  }
}
