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
package org.sonar.plugins.python.api;

import java.util.Set;
import java.util.stream.Collectors;

import static org.sonar.plugins.python.api.PythonVersionUtils.Version;
import static org.sonar.plugins.python.api.PythonVersionUtils.allVersions;

public class ProjectPythonVersion {

  private ProjectPythonVersion() {
  }

  private static Set<Version> currentVersions = allVersions();

  public static Set<Version> currentVersions() {
    return currentVersions;
  }

  public static void setCurrentVersions(Set<Version> currentVersions) {
    ProjectPythonVersion.currentVersions = currentVersions;
  }

  public static Set<String> currentVersionValues() {
    return currentVersions().stream()
      .map(PythonVersionUtils.Version::serializedValue)
      .collect(Collectors.toSet());
  }
}
