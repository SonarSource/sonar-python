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
package org.sonar.plugins.python.api;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_310;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_311;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_312;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_313;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_36;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_37;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_38;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_39;

class PythonVersionUtilsTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  private static final List<PythonVersionUtils.Version> allVersions = List.of(V_36, V_37, V_38, V_39, V_310, V_311, V_312, V_313);

  @Test
  void supportedVersions() {
    assertThat(PythonVersionUtils.fromString("")).hasSameElementsAs(allVersions);
    assertThat(PythonVersionUtils.fromString(",")).hasSameElementsAs(allVersions);
    assertThat(PythonVersionUtils.fromString("2.7")).hasSameElementsAs(allVersions);
    assertThat(PythonVersionUtils.fromString("2")).hasSameElementsAs(allVersions);
    assertThat(PythonVersionUtils.fromString("3")).hasSameElementsAs(allVersions);
    assertThat(PythonVersionUtils.fromString("3.8, 3.9")).containsExactlyInAnyOrder(V_38, V_39);
    assertThat(PythonVersionUtils.fromString("2.7, 3.9")).hasSameElementsAs(allVersions);
    assertThat(PythonVersionUtils.fromString("3.10")).containsExactlyInAnyOrder(V_310);
  }

  @Test
  void version_out_of_range() {
    assertThat(PythonVersionUtils.fromString("4")).containsExactlyInAnyOrder(V_313);
    assertThat(logTester.logs(Level.WARN)).contains("No explicit support for version 4. Python version has been set to 3.13.");
    assertThat(PythonVersionUtils.fromString("1")).hasSameElementsAs(allVersions);
    assertThat(logTester.logs(Level.WARN)).contains("No explicit support for version 1. Support for Python versions prior to 3 is deprecated.");
    assertThat(PythonVersionUtils.fromString("3.14")).containsExactlyInAnyOrder(V_313);
    assertThat(logTester.logs(Level.WARN)).contains("No explicit support for version 3.14. Python version has been set to 3.13.");
    assertThat(PythonVersionUtils.fromString("3.12")).containsExactlyInAnyOrder(V_312);
  }

  @Test
  void bugfix_versions() {
    assertThat(PythonVersionUtils.fromString("3.8.1")).containsExactlyInAnyOrder(V_38);
    assertThat(logTester.logs(Level.WARN)).contains("No explicit support for version 3.8.1. Python version has been set to 3.8.");
    assertThat(PythonVersionUtils.fromString("3.11.1")).containsExactlyInAnyOrder(V_311);
    assertThat(PythonVersionUtils.fromString("3.12.1")).containsExactlyInAnyOrder(V_312);
  }

  @Test
  void error_while_parsing_version() {
    assertThat(PythonVersionUtils.fromString("foo")).hasSameElementsAs(allVersions);
    assertThat(logTester.logs(Level.WARN))
      .contains("Error while parsing value of parameter 'sonar.python.version' (foo). Versions must be specified as MAJOR_VERSION.MINOR_VERSION (e.g. \"3.7, 3.8\")");
  }

  @Test
  void isPythonVersionGreaterOrEqualThan() {
    assertFalse(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(), V_39));
    assertFalse(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_36, V_38), V_39));
    assertFalse(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_36, V_310), V_39));
    assertTrue(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_39), V_39));
    assertTrue(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_39, V_310), V_39));
    assertTrue(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_312, V_310), V_39));
  }
  
}
