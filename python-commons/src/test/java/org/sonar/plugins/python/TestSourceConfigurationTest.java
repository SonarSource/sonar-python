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
package org.sonar.plugins.python;

import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.api.config.Configuration;
import org.sonar.scanner.plugin.api.impl.config.MapSettings;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonarsource.analyzer.commons.appsec.TestFileClassifier.HEURISTIC_DISABLED_KEY;

class TestSourceConfigurationTest {

  @ParameterizedTest(name = "{0}")
  @MethodSource("configurationCases")
  void is_configured(String caseName, Configuration configuration, boolean expected) {
    assertThat(TestSourceConfiguration.isConfigured(configuration)).isEqualTo(expected);
  }

  private static Stream<Arguments> configurationCases() {
    return Stream.of(
      Arguments.of("without test-source configuration", new MapSettings().asConfig(), false),
      Arguments.of("with a blank sonar.tests property", configurationWith("sonar.tests", "  "), false),
      Arguments.of("with sonar.tests configured", configurationWith("sonar.tests", "tests"), true),
      Arguments.of(
        "with the Python heuristic enabled",
        configurationWith("sonar.python.testFileHeuristic.disabled", "false"),
        false),
      Arguments.of(
        "with the Python heuristic disabled",
        configurationWith("sonar.python.testFileHeuristic.disabled", "true"),
        true),
      Arguments.of("with the common heuristic enabled", configurationWith(HEURISTIC_DISABLED_KEY, "false"), false),
      Arguments.of("with the common heuristic disabled", configurationWith(HEURISTIC_DISABLED_KEY, "true"), true)
    );
  }

  private static Configuration configurationWith(String key, String value) {
    return new MapSettings().setProperty(key, value).asConfig();
  }
}
