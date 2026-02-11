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
package org.sonar.plugins.python.telemetry;

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
  PYTHON_NUMBER_OF_FILES_KEY("python.files.count"),
  PYTHON_PACKAGES_WITH_INIT("python.packages.with_init"),
  PYTHON_PACKAGES_WITHOUT_INIT("python.packages.without_init"),
  PYTHON_DUPLICATE_PACKAGES_WITHOUT_INIT("python.packages.duplicate_without_init"),
  PYTHON_NAMESPACE_PACKAGES_IN_REGULAR_PACKAGE("python.packages.namespace_packages_in_regular_package"),
  PYTHON_TYPES_NAMES_TOTAL("python.types.names.total"),
  PYTHON_TYPES_NAMES_UNKNOWN("python.types.names.unknown"),
  PYTHON_TYPES_NAMES_UNRESOLVED_IMPORT("python.types.names.unresolved_import"),
  PYTHON_TYPES_IMPORTS_TOTAL("python.types.imports.total"),
  PYTHON_TYPES_IMPORTS_UNKNOWN("python.types.imports.unknown"),
  PYTHON_TYPES_SYMBOLS_UNIQUE("python.types.symbols.unique"),
  PYTHON_TYPES_SYMBOLS_UNKNOWN("python.types.symbols.unknown"),
  PYTHON_MAIN_FILES_TOTAL("python.files.main.total"),
  PYTHON_MAIN_FILES_MISCLASSIFIED_TEST("python.files.main.misclassified_test"),
  PYTHON_LINES_TOTAL("python.lines.total"),
  PYTHON_MAIN_LINES("python.lines.main"),
  PYTHON_TEST_LINES("python.lines.test"),
  PYTHON_MAIN_LINES_MISCLASSIFIED_TEST("python.lines.main.misclassified_test");

  private final String key;

  TelemetryMetricKey(String key) {
    this.key = key;
  }

  public String key() {
    return key;
  }
}
