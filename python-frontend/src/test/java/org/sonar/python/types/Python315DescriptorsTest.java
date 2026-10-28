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
package org.sonar.python.types;

import java.util.Set;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.PythonVersionUtils.Version;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.typeshed.TypeShedDescriptorsProvider;

import static org.assertj.core.api.Assertions.assertThat;

class Python315DescriptorsTest {
  @AfterEach
  void resetVersions() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @ParameterizedTest
  @EnumSource(value = Version.class, names = {"V_310", "V_311", "V_312", "V_313", "V_314", "V_315"})
  void committed_descriptors_preserve_addition_and_removal_versions(Version version) {
    ProjectPythonVersion.setCurrentVersions(Set.of(version));
    TypeShed.setProjectLevelSymbolTable(ProjectLevelSymbolTable.empty());
    TypeShed.resetBuiltinSymbols();
    var provider = new TypeShedDescriptorsProvider(Set.of(), Set.of(version));
    boolean python315 = version == Version.V_315;
    for (String name : Set.of("frozendict", "sentinel")) {
      assertThat(TypeShed.builtinSymbols().containsKey(name)).as("v1 %s in %s", name, version).isEqualTo(python315);
      assertThat(provider.builtinDescriptors().containsKey(name)).as("v2 %s in %s", name, version).isEqualTo(python315);
    }
    assertThat(TypeShed.typeShedClass("bytearray").resolveMember("take_bytes").isPresent()).isEqualTo(python315);
    var bytearray = (ClassDescriptor) provider.builtinDescriptors().get("bytearray");
    assertThat(bytearray.members().stream().anyMatch(member -> "take_bytes".equals(member.name()))).isEqualTo(python315);
    for (String module : Set.of("math.integer", "profiling", "profiling.sampling")) {
      assertThat(TypeShed.symbolsForModule(module).isEmpty()).as("v1 %s in %s", module, version).isEqualTo(!python315);
      assertThat(provider.descriptorsForModule(module).isEmpty()).as("v2 %s in %s", module, version).isEqualTo(!python315);
    }
    boolean removedLib2to3 = !Set.of(Version.V_310, Version.V_311, Version.V_312).contains(version);
    assertThat(TypeShed.symbolsForModule("lib2to3").isEmpty()).isEqualTo(removedLib2to3);
    assertThat(provider.descriptorsForModule("lib2to3").isEmpty()).isEqualTo(removedLib2to3);
  }
}
