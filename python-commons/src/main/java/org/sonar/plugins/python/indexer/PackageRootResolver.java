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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.sonar.api.config.Configuration;

/**
 * Resolves package root directories for Python projects.
 *
 * <p>This class validates and resolves package roots extracted from build system configurations
 * (e.g., pyproject.toml) and provides fallback resolution when no roots are configured.
 */
public class PackageRootResolver {
  static final String SONAR_SOURCES_KEY = "sonar.sources";
  static final List<String> CONVENTIONAL_FOLDERS = List.of("src", "lib");

  private PackageRootResolver() {
  }

  /**
   * Resolves package root directories.
   *
   * <p>If extracted roots from build system configuration are provided (already as absolute paths
   * resolved relative to their config file locations), returns them directly.
   * Otherwise, applies a fallback chain to determine appropriate roots.
   *
   * @param extractedRoots roots extracted from build system config, already as absolute paths
   * @param config the Sonar configuration to read sonar.sources property
   * @param baseDir the project base directory (used only for fallback resolution)
   * @return list of resolved package root absolute paths
   */
  public static List<String> resolve(List<String> extractedRoots, Configuration config, File baseDir) {
    if (!extractedRoots.isEmpty()) {
      // Extracted roots are already absolute paths (resolved relative to config file location)
      return extractedRoots;
    }
    return resolveFallback(config, baseDir);
  }

  /**
   * Resolves fallback package roots when no build system configuration is available.
   *
   * <p>Fallback priority:
   * <ol>
   *   <li>sonar.sources property if set</li>
   *   <li>"src" and/or "lib" folders if they exist</li>
   *   <li>Project base directory absolute path as last resort</li>
   * </ol>
   *
   * @param config  the Sonar configuration
   * @param baseDir the project base directory
   * @return list of fallback package root absolute paths
   */
  static List<String> resolveFallback(Configuration config, File baseDir) {
    List<String> conventionalFolders = findConventionalFolders(baseDir);
    if (!conventionalFolders.isEmpty()) {
      return toAbsolutePaths(conventionalFolders, baseDir);
    }

    String[] sonarSources = config.getStringArray(SONAR_SOURCES_KEY);
    if (sonarSources.length > 0) {
      return toAbsolutePaths(Arrays.asList(sonarSources), baseDir);
    }

    return List.of(baseDir.getAbsolutePath());
  }

  private static List<String> toAbsolutePaths(List<String> paths, File baseDir) {
    return paths.stream()
      .map(path -> new File(baseDir, path).getAbsolutePath())
      .toList();
  }

  private static List<String> findConventionalFolders(File baseDir) {
    List<String> folders = new ArrayList<>();
    for (String folderName : CONVENTIONAL_FOLDERS) {
      File folder = new File(baseDir, folderName);
      if (folder.exists() && folder.isDirectory()) {
        folders.add(folderName);
      }
    }
    return folders;
  }
}

