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
package org.sonar.python.semantic;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class DependencyGraphTest {

  @Test
  void testDependentModules() {
    Map<String, Set<String>> importsByModule = new HashMap<>();
    importsByModule.put("mod1", Set.of("A", "B"));
    importsByModule.put("mod2", Set.of("B", "C"));
    importsByModule.put("mod3", Set.of("external", "external.other", "B"));
    importsByModule.put("mod4", Set.of("D.E.foo", "F.G.H"));
    Set<String> projectModulesFQN = Set.of("A", "B", "C", "D.E", "F.G.H");
    DependencyGraph dependencyGraph = DependencyGraph.from(importsByModule, projectModulesFQN);

    Map<String, Set<String>> dependentModules = dependencyGraph.dependentModules();

    assertThat(dependentModules).containsOnly(
      Map.entry("A", Set.of("mod1")),
      Map.entry("B", Set.of("mod1", "mod2", "mod3")),
      Map.entry("C", Set.of("mod2")),
      Map.entry("D.E", Set.of("mod4")),
      Map.entry("F.G.H", Set.of("mod4"))
    );
  }

  @Test
  void testModulesToUpdate() {
    Map<String, Set<String>> importsByModule = new HashMap<>();
    importsByModule.put("mod1", Set.of("A", "B", "mod4"));
    importsByModule.put("mod2", Set.of("B", "C"));
    importsByModule.put("mod3", Set.of("external", "B"));
    importsByModule.put("mod4", Set.of("D.E.foo", "F.G.H"));
    Set<String> projectModulesFQN = Set.of("A", "B", "C", "D.E", "F.G.H", "mod4");
    DependencyGraph dependencyGraph = DependencyGraph.from(importsByModule, projectModulesFQN);

    Set<String> strings = dependencyGraph.impactedModules(List.of("D.E"));
    assertThat(strings).containsOnly("mod1", "mod4", "D.E");
  }

  @Test
  void cyclic_dependencies() {
    Map<String, Set<String>> importsByModule = new HashMap<>();
    importsByModule.put("mod1", Set.of("mod2"));
    importsByModule.put("mod2", Set.of("mod1"));
    Set<String> projectModulesFQN = Set.of("mod1", "mod2");
    DependencyGraph dependencyGraph = DependencyGraph.from(importsByModule, projectModulesFQN);

    Set<String> strings = dependencyGraph.impactedModules(List.of("mod1"));
    assertThat(strings).containsOnly("mod1", "mod2");
  }
}
