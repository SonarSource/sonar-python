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

import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.PythonVersionUtils.SemanticVersion;
import org.sonar.plugins.python.api.PythonVersionUtils.Version;

import static org.sonar.plugins.python.api.PythonVersionUtils.allVersions;

public class ProjectPythonVersion {

  private ProjectPythonVersion() {
  }

  private static Set<Version> currentVersions = allVersions();

  /**
   * Returns the normalized source-version compatibility buckets derived from project configuration.
   * Version-sensitive rules should use this set.
   */
  public static Set<Version> currentVersions() {
    return currentVersions;
  }

  /**
   * Sets the normalized source-version compatibility buckets for the current project.
   */
  public static void setCurrentVersions(Set<Version> currentVersions) {
    ProjectPythonVersion.currentVersions = currentVersions;
  }

  /**
   * Returns the serialized identifiers of the semantic models selected for the current project.
   *
   * @deprecated Use {@link #currentSemanticVersions()} to keep semantic versions strongly typed.
   */
  @Deprecated(since = "5.31")
  public static Set<String> currentVersionValues() {
    return currentSemanticVersions().stream()
      .map(SemanticVersion::serializedValue)
      .collect(Collectors.toSet());
  }

  /**
   * Returns the versions used to select serialized semantic-model data. Legacy source-version compatibility buckets
   * are mapped to the oldest semantic version for which data is generated.
   */
  public static Set<SemanticVersion> currentSemanticVersions() {
    return PythonVersionUtils.toSupportedSemanticVersions(currentVersions());
  }
}
