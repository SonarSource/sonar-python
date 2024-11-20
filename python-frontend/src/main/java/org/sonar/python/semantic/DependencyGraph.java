/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DependencyGraph {

  private final Map<String, Set<String>> dependentModules;

  private DependencyGraph(Map<String, Set<String>> dependentModules) {
    this.dependentModules = dependentModules;
  }

  public Map<String, Set<String>> dependentModules() {
    return Collections.unmodifiableMap(dependentModules);
  }

  public static DependencyGraph from(Map<String, Set<String>> importsByModule, Set<String> projectModulesFQN) {
    Map<String, Set<String>> dependentModules = computeDependentModules(importsByModule, projectModulesFQN);
    return new DependencyGraph(dependentModules);
  }

  private static Map<String, Set<String>> computeDependentModules(Map<String, Set<String>> importsByModule, Set<String> projectModulesFQN) {
    Map<String, Set<String>> result = new HashMap<>();
    for (var entry : importsByModule.entrySet()) {
      entry.getValue().forEach(importedModuleFQN -> {
        String dependentModule = entry.getKey();
        if (projectModulesFQN.contains(importedModuleFQN)) {
          result.computeIfAbsent(importedModuleFQN, x -> new HashSet<>()).add(dependentModule);
          return;
        }
        int endIndex = importedModuleFQN.lastIndexOf(".");
        if (endIndex < 0) {
          return;
        }
        String substring = importedModuleFQN.substring(0, endIndex);
        if (projectModulesFQN.contains(substring)) {
          result.computeIfAbsent(substring, x -> new HashSet<>()).add(dependentModule);
        }
      });
    }
    return result;
  }

  public Set<String> impactedModules(List<String> modifiedModules) {
    Set<String> impactedModules = new HashSet<>();
    for (String modifiedModuleFQN : modifiedModules) {
      recursivelyComputeImpactedModules(modifiedModuleFQN, impactedModules);
    }
    return impactedModules;
  }

  private void recursivelyComputeImpactedModules(String changedModule, Set<String> impactedModules) {
    if (!impactedModules.contains(changedModule)) {
      impactedModules.add(changedModule);
      Set<String> transitivelyImpactedModules = dependentModules.get(changedModule);
      if (transitivelyImpactedModules == null) {
        return;
      }
      for (String transitivelyImpacted : transitivelyImpactedModules) {
        recursivelyComputeImpactedModules(transitivelyImpacted, impactedModules);
      }
    }
  }
}
