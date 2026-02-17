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

import javax.annotation.Nullable;

/**
 * Telemetry data for namespace packages.
 * 
 * @param packagesWithInit The number of packages with an __init__.py file.
 * @param packagesWithoutInit The number of packages without an __init__.py file.
 * @param duplicatePackagesWithoutInit The number of packages without an __init__.py file that appear multiple times. Each occurrence increments this count.
 * @param namespacePackagesInRegularPackage The number of packages without an __init__.py file that have at least one parent with an __init__.py file.
 * @param resolutionMethod How the package roots were resolved (e.g., pyproject.toml, setup.py, sonar.sources, fallback).
 * @param buildSystem The build system identified when resolution method is PYPROJECT_TOML (e.g., setuptools, poetry).
 */
public record NamespacePackageTelemetry(
  int packagesWithInit,
  int packagesWithoutInit,
  int duplicatePackagesWithoutInit,
  int namespacePackagesInRegularPackage,
  @Nullable PackageResolutionResult.ResolutionMethod resolutionMethod,
  @Nullable PackageResolutionResult.BuildSystem buildSystem) {
  
  public static NamespacePackageTelemetry empty() {
    return new NamespacePackageTelemetry(0, 0, 0, 0, null, null);
  }

  /**
   * Creates a new telemetry instance with resolution information added.
   */
  public NamespacePackageTelemetry withResolutionInfo(
      PackageResolutionResult.ResolutionMethod method,
      PackageResolutionResult.BuildSystem system) {
    return new NamespacePackageTelemetry(
      packagesWithInit,
      packagesWithoutInit,
      duplicatePackagesWithoutInit,
      namespacePackagesInRegularPackage,
      method,
      system);
  }
}

