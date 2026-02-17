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

import java.util.List;

/**
 * Result of package root resolution containing both the resolved roots and
 * information about how they were resolved.
 *
 * @param roots The resolved package root absolute paths
 * @param method The method used to resolve the package roots
 * @param buildSystem The build system identified (only applicable for PYPROJECT_TOML method)
 */
public record PackageResolutionResult(
  List<String> roots,
  ResolutionMethod method,
  BuildSystem buildSystem) {

  /**
   * How the package roots were resolved.
   */
  public enum ResolutionMethod {
    /** Resolved from pyproject.toml build configuration */
    PYPROJECT_TOML,
    /** Resolved from setup.py configuration */
    SETUP_PY,
    /** Resolved from both pyproject.toml and setup.py */
    PYPROJECT_AND_SETUP_PY,
    /** Resolved from sonar.sources property */
    SONAR_SOURCES,
    /** Resolved from conventional folders (src/, lib/) */
    CONVENTIONAL_FOLDERS,
    /** Fallback to project base directory */
    BASE_DIR
  }

  /**
   * Build systems supported in pyproject.toml.
   */
  public enum BuildSystem {
    SETUPTOOLS,
    POETRY,
    HATCHLING,
    UV_BUILD,
    UV_BUILD_DEFAULT_MODULE,
    PDM,
    FLIT,
    /** Multiple build systems detected */
    MULTIPLE,
    /** Used when resolution method is not PYPROJECT_TOML */
    NONE
  }

  public static PackageResolutionResult fromPyProjectToml(List<String> roots, BuildSystem buildSystem) {
    return new PackageResolutionResult(roots, ResolutionMethod.PYPROJECT_TOML, buildSystem);
  }

  public static PackageResolutionResult fromSetupPy(List<String> roots) {
    return new PackageResolutionResult(roots, ResolutionMethod.SETUP_PY, BuildSystem.NONE);
  }

  public static PackageResolutionResult fromBothPyProjectAndSetupPy(List<String> roots, BuildSystem buildSystem) {
    return new PackageResolutionResult(roots, ResolutionMethod.PYPROJECT_AND_SETUP_PY, buildSystem);
  }

  public static PackageResolutionResult fromSonarSources(List<String> roots) {
    return new PackageResolutionResult(roots, ResolutionMethod.SONAR_SOURCES, BuildSystem.NONE);
  }

  public static PackageResolutionResult fromConventionalFolders(List<String> roots) {
    return new PackageResolutionResult(roots, ResolutionMethod.CONVENTIONAL_FOLDERS, BuildSystem.NONE);
  }

  public static PackageResolutionResult fromBaseDir(List<String> roots) {
    return new PackageResolutionResult(roots, ResolutionMethod.BASE_DIR, BuildSystem.NONE);
  }
}
