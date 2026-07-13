/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.telemetry.collectors;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.assertj.core.api.Assertions.assertThat;

class ImportsTelemetryCollectorTest {

  private static PythonVisitorContext createContext(String code) {
    var mockFile = new TestPythonVisitorRunner.MockPythonFile("", "test.py", code);
    return TestPythonVisitorRunner.createContext(mockFile, null, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());
  }

  private static ImportsTelemetry collectFrom(String code) {
    var collector = new ImportsTelemetryCollector();
    collector.collect(createContext(code).rootTree());
    return collector.getTelemetry();
  }

  @Test
  void simpleImport() {
    var telemetry = collectFrom("import pandas");
    assertThat(telemetry.importedModules()).containsExactly("pandas");
  }

  @Test
  void dottedImportRecordsOnlyTopLevel() {
    var telemetry = collectFrom("import os.path");
    assertThat(telemetry.importedModules()).containsExactly("os");
  }

  @Test
  void fromImportRecordsModule() {
    var telemetry = collectFrom("from pandas import DataFrame");
    assertThat(telemetry.importedModules()).containsExactly("pandas");
  }

  @Test
  void fromDottedImportRecordsOnlyTopLevel() {
    var telemetry = collectFrom("from os.path import join");
    assertThat(telemetry.importedModules()).containsExactly("os");
  }

  @Test
  void wildcardImportRecordsModule() {
    var telemetry = collectFrom("from os import *");
    assertThat(telemetry.importedModules()).containsExactly("os");
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "from . import foo",
    "from ..utils import bar",
    "",
    "x = 1\nprint(x)"
  })
  void producesEmptyImportedModules(String code) {
    var telemetry = collectFrom(code);
    assertThat(telemetry.importedModules()).isEmpty();
  }

  @Test
  void multipleModulesOnOneImportStatement() {
    var telemetry = collectFrom("import os, sys");
    assertThat(telemetry.importedModules()).containsExactlyInAnyOrder("os", "sys");
  }

  @Test
  void duplicatesAreDeduplicatedAcrossFiles() {
    var collector = new ImportsTelemetryCollector();
    collector.collect(createContext("import pandas").rootTree());
    collector.collect(createContext("import pandas\nimport numpy").rootTree());
    var telemetry = collector.getTelemetry();
    assertThat(telemetry.importedModules()).containsExactlyInAnyOrder("pandas", "numpy");
  }

  @Test
  void namesAreNormalized() {
    // Module names with uppercase are lowercased
    var telemetry = collectFrom("import Pandas");
    assertThat(telemetry.importedModules()).containsExactly("pandas");
  }

  @Test
  void nameLongerThan100CharsIsFiltered() {
    String longName = "a".repeat(100);
    var telemetry = collectFrom("import " + longName);
    assertThat(telemetry.importedModules()).isEmpty();
  }

  @Test
  void nameShorterThan100CharsIsKept() {
    String shortName = "a".repeat(99);
    var telemetry = collectFrom("import " + shortName);
    assertThat(telemetry.importedModules()).containsExactly(shortName);
  }

  @Test
  void mixOfImportTypes() {
    var telemetry = collectFrom("""
      import os
      import os.path
      from pandas import DataFrame
      from numpy.linalg import norm
      from . import local_module
      from ..utils import helper
      from sys import *
      """);
    assertThat(telemetry.importedModules()).containsExactlyInAnyOrder("os", "pandas", "numpy", "sys");
  }
}
