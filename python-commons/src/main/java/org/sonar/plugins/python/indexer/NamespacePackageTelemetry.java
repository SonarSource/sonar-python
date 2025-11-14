/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

/**
 * Telemetry data for namespace packages.
 * 
 * @param packagesWithInit The number of packages with an __init__.py file.
 * @param packagesWithoutInit The number of packages without an __init__.py file.
 * @param duplicatePackagesWithoutInit The number of packages without an __init__.py file that appear multiple times. Each occurrence increments this count.
 * @param namespacePackagesInRegularPackage The number of packages without an __init__.py file that have at least one parent with an __init__.py file.
 */
public record NamespacePackageTelemetry(
  int packagesWithInit,
  int packagesWithoutInit,
  int duplicatePackagesWithoutInit,
  int namespacePackagesInRegularPackage) {
  
  public static NamespacePackageTelemetry empty() {
    return new NamespacePackageTelemetry(0, 0, 0, 0);
  }
}

