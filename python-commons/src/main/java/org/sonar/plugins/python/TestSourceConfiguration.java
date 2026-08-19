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

import org.sonar.api.config.Configuration;

import static org.sonarsource.analyzer.commons.appsec.TestFileClassifier.HEURISTIC_DISABLED_KEY;

/**
 * Resolves whether explicit test-source classification is authoritative for the analysis,
 * in which case the test-content heuristic must not be applied.
 */
public final class TestSourceConfiguration {

  private TestSourceConfiguration() {
  }

  /**
   * Checks whether test source classification is explicitly configured.
   * @param config analysis configuration
   * @return whether the heuristic must be disabled
   */
  public static boolean isConfigured(Configuration config) {
    return isPropertyConfigured(config, "sonar.tests")
      || config.getBoolean("sonar.python.testFileHeuristic.disabled").orElse(false)
      || config.getBoolean(HEURISTIC_DISABLED_KEY).orElse(false);
  }

  private static boolean isPropertyConfigured(Configuration config, String key) {
    return config.get(key).filter(value -> !value.isBlank()).isPresent();
  }
}
