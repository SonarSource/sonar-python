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
import java.util.Map;
import java.util.stream.Collectors;

public class NamespacePackageAnalyzer {

  public NamespacePackageTelemetry analyze(ProjectTree projectTree) {
    List<ProjectTreeFolder> foldersWithPythonFiles = projectTree.allFolders()
      .filter(folder -> !"/".equals(folder.name()))
      .filter(NamespacePackageAnalyzer::hasPythonFiles)
      .toList();

    int packagesWithInit = 0;
    int packagesWithoutInit = 0;
    int duplicatePackagesWithoutInit = 0;
    int namespacePackagesInRegularPackage = 0;

    Map<String, Long> folderNameCounts = foldersWithPythonFiles.stream()
      .collect(Collectors.groupingBy(ProjectTreeFolder::name, Collectors.counting()));

    for (ProjectTreeFolder folder : foldersWithPythonFiles) {
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
      namespacePackagesInRegularPackage,
      null,
      null);
  }

  private static boolean hasAnyParentWithInit(ProjectTreeFolder folder) {
    return folder.parents().anyMatch(NamespacePackageAnalyzer::hasInitFile);
  }

  private static boolean hasInitFile(ProjectTreeFolder folder) {
    return folder.children().stream()
      .anyMatch(child -> child instanceof ProjectTreeFile && "__init__.py".equals(child.name()));
  }

  private static boolean hasPythonFiles(ProjectTreeFolder folder) {
    return folder.children().stream()
      .anyMatch(child -> child instanceof ProjectTreeFile && child.name().endsWith(".py"));
  }
}

