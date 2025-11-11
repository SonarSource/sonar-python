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
package org.sonar.plugins.python.indexer;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class NamespacePackageAnalyzer {

  public NamespacePackageTelemetry analyze(ProjectTree projectTree) {
    List<ProjectTree.ProjectTreeFolder> foldersWithPythonFiles = projectTree.allFolders()
      .filter(folder -> !"/".equals(folder.name()))
      .filter(NamespacePackageAnalyzer::hasPythonFiles)
      .toList();

    int packagesWithInit = 0;
    int packagesWithoutInit = 0;
    int duplicatePackagesWithoutInit = 0;
    int namespacePackagesInRegularPackage = 0;

    Map<String, Long> folderNameCounts = foldersWithPythonFiles.stream()
      .collect(Collectors.groupingBy(ProjectTree.ProjectTreeFolder::name, Collectors.counting()));

    for (ProjectTree.ProjectTreeFolder folder : foldersWithPythonFiles) {
      if (hasInitFile(folder)) {
        packagesWithInit++;
      } else {
        packagesWithoutInit++;

        boolean appearMultipleTimes = folderNameCounts.get(folder.name()) > 1;
        if (appearMultipleTimes) {
          duplicatePackagesWithoutInit++;
        }

        if (hasAnyParentWithInit(folder)) {
          namespacePackagesInRegularPackage++;
        }
      }
    }

    return new NamespacePackageTelemetry(
      packagesWithInit,
      packagesWithoutInit,
      duplicatePackagesWithoutInit,
      namespacePackagesInRegularPackage);
  }

  private static boolean hasAnyParentWithInit(ProjectTree.ProjectTreeFolder folder) {
    return folder.parents().anyMatch(NamespacePackageAnalyzer::hasInitFile);
  }

  private static boolean hasInitFile(ProjectTree.ProjectTreeFolder folder) {
    return folder.children().stream()
      .anyMatch(child -> child instanceof ProjectTree.ProjectTreeFile && "__init__.py".equals(child.name()));
  }

  private static boolean hasPythonFiles(ProjectTree.ProjectTreeFolder folder) {
    return folder.children().stream()
      .anyMatch(child -> child instanceof ProjectTree.ProjectTreeFile && child.name().endsWith(".py"));
  }
}

