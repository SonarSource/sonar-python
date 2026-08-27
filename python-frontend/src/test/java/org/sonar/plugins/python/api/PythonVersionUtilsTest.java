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
package org.sonar.plugins.python.api;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.event.Level;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_310;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_311;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_312;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_313;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_314;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_38;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_39;

class PythonVersionUtilsTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  private static final List<PythonVersionUtils.Version> supportedVersions = List.of(V_310, V_311, V_312, V_313, V_314);
  private static final List<PythonVersionUtils.Version> recognizedVersions = List.of(V_38, V_39, V_310, V_311, V_312, V_313, V_314);

  @Test
  void supportedVersions() {
    assertThat(PythonVersionUtils.fromString("")).hasSameElementsAs(supportedVersions);
    assertThat(PythonVersionUtils.fromString(",")).hasSameElementsAs(supportedVersions);
    assertThat(PythonVersionUtils.fromString("3")).hasSameElementsAs(supportedVersions);
    assertThat(PythonVersionUtils.fromString("3.10, 3.12, 3.14")).containsExactlyInAnyOrder(V_310, V_312, V_314);
    assertThat(PythonVersionUtils.fromString("3.10")).containsExactlyInAnyOrder(V_310);
  }

  @Test
  void legacy_python_versions_are_preserved_for_rule_policy() {
    assertThat(PythonVersionUtils.fromString("2.7")).containsExactly(V_38);
    assertThat(PythonVersionUtils.fromString("2")).containsExactly(V_38);
    assertThat(PythonVersionUtils.fromString("2.7, 3.10")).containsExactly(V_38, V_310);
    assertThat(PythonVersionUtils.fromString("3.8")).containsExactly(V_38);
    assertThat(PythonVersionUtils.fromString("3.9")).containsExactly(V_39);
  }

  @Test
  void legacy_python_3_versions_use_minimum_supported_semantic_version() {
    assertThat(PythonVersionUtils.toSupportedSemanticVersions(Set.of(V_38)))
      .containsExactly(PythonVersionUtils.SemanticVersion.V_310);
    assertThat(PythonVersionUtils.toSupportedSemanticVersions(Set.of(V_39, V_310)))
      .containsExactly(PythonVersionUtils.SemanticVersion.V_310);
    assertThat(PythonVersionUtils.toSupportedSemanticVersions(Set.of(V_311, V_314)))
      .containsExactly(PythonVersionUtils.SemanticVersion.V_311, PythonVersionUtils.SemanticVersion.V_314);
  }

  @Test
  void all_supported_semantic_versions_exclude_recognized_legacy_source_versions() {
    assertThat(PythonVersionUtils.allSupportedSemanticVersions()).containsExactly(
      PythonVersionUtils.SemanticVersion.V_310,
      PythonVersionUtils.SemanticVersion.V_311,
      PythonVersionUtils.SemanticVersion.V_312,
      PythonVersionUtils.SemanticVersion.V_313,
      PythonVersionUtils.SemanticVersion.V_314);
    assertThat(PythonVersionUtils.allSupportedSemanticVersions())
      .extracting(PythonVersionUtils.SemanticVersion::serializedValue)
      .containsExactly("310", "311", "312", "313", "314");
  }

  @Test
  void supported_semantic_version_bounds() {
    assertThat(PythonVersionUtils.minSupportedSemanticVersion())
      .isEqualTo(PythonVersionUtils.SemanticVersion.V_310)
      .hasToString("3.10");
    assertThat(PythonVersionUtils.maxSupportedSemanticVersion())
      .isEqualTo(PythonVersionUtils.SemanticVersion.V_314)
      .hasToString("3.14");
  }

  @Test
  @SuppressWarnings("deprecation")
  void deprecated_serialized_source_version_value_is_preserved_for_compatibility() {
    assertThat(V_38.serializedValue()).isEqualTo("38");
    assertThat(V_310.serializedValue()).isEqualTo("310");
  }

  @Test
  void version_out_of_range() {
    assertThat(PythonVersionUtils.fromString("4")).containsExactlyInAnyOrder(V_314);
    assertThat(logTester.logs(Level.WARN)).contains("No explicit support for version 4. Python version has been set to 3.14.");
    assertThat(PythonVersionUtils.fromString("1")).containsExactly(V_38);
    assertThat(logTester.logs(Level.WARN)).contains("No explicit support for version 1. Support for Python versions prior to 3 is deprecated.");
    assertThat(PythonVersionUtils.fromString("3.15")).containsExactlyInAnyOrder(V_314);
    assertThat(logTester.logs(Level.WARN)).contains("No explicit support for version 3.15. Python version has been set to 3.14.");
    assertThat(PythonVersionUtils.fromString("3.12")).containsExactlyInAnyOrder(V_312);
  }

  @Test
  void bugfix_versions() {
    assertThat(PythonVersionUtils.fromString("3.9.1")).containsExactlyInAnyOrder(V_39);
    assertThat(logTester.logs(Level.WARN)).contains("No explicit support for version 3.9.1. Python version has been set to 3.9.");
    assertThat(PythonVersionUtils.fromString("3.11.1")).containsExactlyInAnyOrder(V_311);
    assertThat(PythonVersionUtils.fromString("3.12.1")).containsExactlyInAnyOrder(V_312);
  }

  @Test
  void comparison_specifiers() {
    assertThat(PythonVersionUtils.fromString(">=3.10")).containsExactly(V_310, V_311, V_312, V_313, V_314);
    assertThat(PythonVersionUtils.fromString(">3.10")).containsExactly(V_311, V_312, V_313, V_314);
    assertThat(PythonVersionUtils.fromString(">=3,<=3.10")).containsExactly(V_38, V_39, V_310);
    assertThat(PythonVersionUtils.fromString(">=3,<3.10")).containsExactly(V_38, V_39);
    assertThat(PythonVersionUtils.fromString(">=3.10.1,<3.12.4")).containsExactly(V_310, V_311);
    assertThat(PythonVersionUtils.fromString("<3.12")).containsExactly(V_38, V_39, V_310, V_311);
    assertThat(PythonVersionUtils.fromString(">=2.7,<3.12")).containsExactly(V_38, V_39, V_310, V_311);
    assertThat(PythonVersionUtils.fromString(" >= 3.9 , < 3.12 ")).containsExactly(V_39, V_310, V_311);
  }

  @Test
  void equality_and_exclusion_specifiers() {
    assertThat(PythonVersionUtils.fromString("==3.11")).containsExactly(V_311);
    assertThat(PythonVersionUtils.fromString("==3.*")).hasSameElementsAs(recognizedVersions);
    assertThat(PythonVersionUtils.fromString("==3.11.*")).containsExactly(V_311);
    assertThat(PythonVersionUtils.fromString("==3.11.4")).containsExactly(V_311);
    assertThat(PythonVersionUtils.fromString("!=3.11")).containsExactly(V_38, V_39, V_310, V_312, V_313, V_314);
    assertThat(PythonVersionUtils.fromString(">=3,!=3.11")).containsExactly(V_38, V_39, V_310, V_312, V_313, V_314);
    assertThat(PythonVersionUtils.fromString(">=3,!=3.11.*")).containsExactly(V_38, V_39, V_310, V_312, V_313, V_314);
  }

  @Test
  void compatible_and_poetry_specifiers() {
    assertThat(PythonVersionUtils.fromString("~=3.10")).containsExactly(V_310, V_311, V_312, V_313, V_314);
    assertThat(PythonVersionUtils.fromString("~=3.10.1")).containsExactly(V_310);
    assertThat(PythonVersionUtils.fromString("^3.10")).containsExactly(V_310, V_311, V_312, V_313, V_314);
    assertThat(PythonVersionUtils.fromString("^3.10.1")).containsExactly(V_310, V_311, V_312, V_313, V_314);
  }

  @Test
  void ranges_without_supported_versions_fall_back_to_all_versions() {
    assertThat(PythonVersionUtils.fromString(">=3.15")).hasSameElementsAs(supportedVersions);
    assertThat(logTester.logs(Level.WARN))
      .contains("No supported Python version matches version range >=3.15. Analysis will target all supported Python versions.");

    assertThat(PythonVersionUtils.fromString(">=3,<3.8")).hasSameElementsAs(supportedVersions);
    assertThat(logTester.logs(Level.WARN))
      .contains("No supported Python version matches version range >=3,<3.8. Analysis will target all supported Python versions.");
  }

  @Test
  void contradictory_or_unsupported_ranges_fall_back_to_all_versions() {
    assertThat(PythonVersionUtils.fromString(">=3.12,<3.11")).hasSameElementsAs(supportedVersions);
    assertThat(logTester.logs(Level.WARN))
      .contains("No supported Python version matches version range >=3.12,<3.11. Analysis will target all supported Python versions.");

    assertThat(PythonVersionUtils.fromString("<3")).hasSameElementsAs(supportedVersions);
    assertThat(logTester.logs(Level.WARN))
      .contains("No supported Python version matches version range <3. Analysis will target all supported Python versions.");
  }

  @ParameterizedTest
  @ValueSource(strings = {"3.10,>=3.11", "~=3", "~3.10", "===3.10", ">=3.10rc1", "^3.10 || ^3.11", ">=3.10.*", "==foo"})
  void unsupported_range_syntax(String value) {
    assertThat(PythonVersionUtils.fromString(value)).hasSameElementsAs(supportedVersions);
    assertThat(logTester.logs(Level.WARN))
      .contains("Error while parsing value of parameter 'sonar.python.version' (" + value
        + "). Use comma-separated Python versions (e.g. \"3.10,3.11\") or numeric version specifiers (e.g. \">=3.10,<3.13\").");
  }

  @Test
  void error_while_parsing_version() {
    assertThat(PythonVersionUtils.fromString("foo")).hasSameElementsAs(supportedVersions);
    assertThat(logTester.logs(Level.WARN))
      .contains("Error while parsing value of parameter 'sonar.python.version' (foo). Use comma-separated Python versions (e.g. \"3.10,3.11\") or numeric version specifiers (e.g. \">=3.10,<3.13\").");
  }

  @Test
  void isPythonVersionGreaterOrEqualThan() {
    assertFalse(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(), V_39));
    assertFalse(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_311, V_312), V_313));
    assertFalse(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_39, V_310), V_310));
    assertFalse(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_310, V_312), V_311));
    assertFalse(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_313, V_312), V_314));
    assertTrue(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_39), V_39));
    assertTrue(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_39, V_310), V_39));
    assertTrue(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_312, V_310), V_39));
    assertTrue(PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(Set.of(V_314), V_314));
  }
  
}
