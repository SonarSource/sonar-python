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

import java.util.Collections;
import java.util.Set;

/**
 * Telemetry data for top-level module names found in Python import statements.
 *
 * @param importedModules Set of normalized top-level module names (e.g. "pandas", "numpy")
 */
public record ImportsTelemetry(Set<String> importedModules) {

  public ImportsTelemetry(Set<String> importedModules) {
    this.importedModules = Collections.unmodifiableSet(importedModules);
  }

}
