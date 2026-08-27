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
  PrimaryResolutionMethod method,
  BuildSystem buildSystem) {

  /**
   * How the package roots were resolved.
   */
  public enum PrimaryResolutionMethod {
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
    BASE_DIR,
    /** No build config files found; roots are computed by scanning __init__.py chains in source files */
    LEGACY_INIT_PY,

    /** Roots were provided by A3S context */
    A3S_CONTEXT
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
    return new PackageResolutionResult(roots, PrimaryResolutionMethod.PYPROJECT_TOML, buildSystem);
  }

  public static PackageResolutionResult fromSetupPy(List<String> roots) {
    return new PackageResolutionResult(roots, PrimaryResolutionMethod.SETUP_PY, BuildSystem.NONE);
  }

  public static PackageResolutionResult fromBothPyProjectAndSetupPy(List<String> roots, BuildSystem buildSystem) {
    return new PackageResolutionResult(roots, PrimaryResolutionMethod.PYPROJECT_AND_SETUP_PY, buildSystem);
  }

  public static PackageResolutionResult fromSonarSources(List<String> roots) {
    return new PackageResolutionResult(roots, PrimaryResolutionMethod.SONAR_SOURCES, BuildSystem.NONE);
  }

  public static PackageResolutionResult fromConventionalFolders(List<String> roots) {
    return new PackageResolutionResult(roots, PrimaryResolutionMethod.CONVENTIONAL_FOLDERS, BuildSystem.NONE);
  }

  public static PackageResolutionResult fromBaseDir(List<String> roots) {
    return new PackageResolutionResult(roots, PrimaryResolutionMethod.BASE_DIR, BuildSystem.NONE);
  }

  public static PackageResolutionResult fromLegacyInitPy(List<String> roots) {
    return new PackageResolutionResult(roots, PrimaryResolutionMethod.LEGACY_INIT_PY, BuildSystem.NONE);
  }

  public static PackageResolutionResult fromA3SContext(List<String> roots) {
    return new PackageResolutionResult(roots, PrimaryResolutionMethod.A3S_CONTEXT, BuildSystem.NONE);
  }
}
