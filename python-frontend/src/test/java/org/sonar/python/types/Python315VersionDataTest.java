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
import org.sonar.python.index.AliasDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.typeshed.TypeShedDescriptorsProvider;

import static org.assertj.core.api.Assertions.assertThat;

class Python315VersionDataTest {
  @AfterEach
  void resetVersions() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @ParameterizedTest
  @EnumSource(value = Version.class, names = {"V_310", "V_311", "V_312", "V_313", "V_314", "V_315"})
  void existing_versioned_api_remains_available_in_python_315(Version version) {
    ProjectPythonVersion.setCurrentVersions(Set.of(version));
    TypeShed.setProjectLevelSymbolTable(ProjectLevelSymbolTable.empty());
    TypeShed.resetBuiltinSymbols();
    var provider = new TypeShedDescriptorsProvider(Set.of(), Set.of(version));
    boolean hasUtc = version != Version.V_310;
    assertThat(TypeShed.symbolsForModule("datetime").containsKey("UTC")).isEqualTo(hasUtc);
    assertThat(provider.descriptorsForModule("datetime").containsKey("UTC")).isEqualTo(hasUtc);
    assertThat(TypeShed.symbolsForModule("datetime")).containsKey("datetime");
    assertThat(provider.descriptorsForModule("datetime")).containsKey("datetime");
  }

  @ParameterizedTest
  @EnumSource(value = Version.class, names = {"V_310", "V_311", "V_312", "V_313", "V_314", "V_315"})
  void regenerated_third_party_descriptors_preserve_concrete_types(Version version) {
    var provider = new TypeShedDescriptorsProvider(Set.of(), Set.of(version));
    var numericColumn = (FunctionDescriptor) originalDescriptor(provider.descriptorsForModule("tensorflow.feature_column").get("numeric_column"));
    assertThat(numericColumn.annotatedReturnTypeName()).isEqualTo("tensorflow.python.feature_column.feature_column_v2.NumericColumn");
    var pixmap = (FunctionDescriptor) originalDescriptor(provider.descriptorsForModule("Xlib.ext.composite").get("name_window_pixmap"));
    assertThat(pixmap.annotatedReturnTypeName()).isEqualTo("Xlib.xobject.drawable.Pixmap");
    var integer = (ClassDescriptor) originalDescriptor(provider.descriptorsForModule("pyasn1.type.univ").get("Integer"));
    assertThat(integer.superClasses()).containsExactly("pyasn1.type.base.SimpleAsn1Type");
  }

  private static Descriptor originalDescriptor(Descriptor descriptor) {
    return descriptor instanceof AliasDescriptor alias ? alias.originalDescriptor() : descriptor;
  }
}
