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
package org.sonar.plugins.python.indexer;

import java.io.File;
import java.util.List;

/**
 * Represents source roots extracted from a build configuration file (pyproject.toml or setup.py).
 *
 * <p>This record associates relative source root paths with the configuration file they were
 * extracted from, allowing proper resolution of absolute paths relative to the config file's
 * location rather than the project base directory.
 *
 * @param configFile the configuration file from which the source roots were extracted
 * @param relativeRoots the relative source root paths defined in the config file
 */
public record ConfigSourceRoots(File configFile, List<String> relativeRoots) {

  /**
   * Converts the relative source roots to absolute paths.
   *
   * <p>Each relative path is resolved relative to the parent directory of the configuration file.
   * For example, if the config file is at {@code /project/app/pyproject.toml} and a relative root
   * is {@code "src"}, the absolute path will be {@code /project/app/src}.
   *
   * @return list of absolute paths for the source roots
   */
  public List<String> toAbsolutePaths() {
    File parentDir = configFile.getParentFile();
    if (parentDir == null) {
      return List.of();
    }
    return relativeRoots.stream()
      .map(root -> new File(parentDir, root).getAbsolutePath())
      .toList();
  }

  /**
   * Creates a ConfigSourceRoots with an empty list of relative roots.
   *
   * @param configFile the configuration file
   * @return a ConfigSourceRoots with no relative roots
   */
  public static ConfigSourceRoots empty(File configFile) {
    return new ConfigSourceRoots(configFile, List.of());
  }
}
