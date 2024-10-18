/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.types.v2;

import java.util.List;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.types.v2.UnknownType.UnresolvedImportType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;


class UnresolvedImportTypeTest {
  @Test
  @Disabled("SONARPY-2213 unknown.submodule is not correctly resolved")
  void imported_unknown_submodule() {
    FileInput fileInput = inferTypesWithNoResolution("""
      import unknown.submodule
      unknown.submodule
      """);
    var unknownSubmoduleType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    assertThat(unknownSubmoduleType.importPath()).isEqualTo("unknown.submodule");
  }


  @Test
  void imported_unknown() {
    FileInput fileInput = inferTypesWithNoResolution("""
      import unknown
      unknown
      """);
    var unknownType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    assertThat(unknownType.importPath()).isEqualTo("unknown");
  }

  @Test
  void imported_unknown2() {
    FileInput fileInput = inferTypesWithNoResolution("""
      import xml as a
      a
      """);
    var etreeType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    assertThat(etreeType.importPath()).isEqualTo("xml");
  }

  @Test
  void imported_flow_union() {
    FileInput fileInput = inferTypesWithNoResolution("""
      from something import acos, atan
      if cond:
        x = acos
      else:
        x = atan
      x
      acos
      atan
      """);
    var xType = ((UnionType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2());
    var acosType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0).typeV2());
    var atanType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0).typeV2());

    assertThat(xType.candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(acosType, atanType);
    assertThat(acosType.importPath()).isEqualTo("something.acos");
    assertThat(atanType.importPath()).isEqualTo("something.atan");
  }

  @Test
  @Disabled("SONARPY-2216 Type of alias from import is not propagated")
  void import_from_as() {
    FileInput fileInput = inferTypesWithNoResolution("""
      from something import a_func as f
      f
      """);
    var aliasType = ((ImportFrom) fileInput.statements().statements().get(0)).importedNames().get(0).alias().typeV2();
    var fType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());

    assertThat(aliasType).isSameAs(fType);
    assertThat(fType).extracting(UnresolvedImportType::importPath).isEqualTo("something.a_func");
  }

  @Test
  void imported_call() {
    FileInput fileInput = inferTypesWithNoResolution("""
      from something import a_func
      x = a_func(1)
      x
      """);
    var xType = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(xType).isInstanceOf(UnknownType.class);
  }

  private static FileInput inferTypesWithNoResolution(String lines) {
    var typeTable = new TestProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    return parseAndInferTypes(typeTable, PythonTestUtils.pythonFile(""), lines);
  }

  private static class TestProjectLevelTypeTable extends ProjectLevelTypeTable {

    public TestProjectLevelTypeTable(ProjectLevelSymbolTable projectLevelSymbolTable) {
      super(projectLevelSymbolTable);
    }

    @Override
    public PythonType getType(List<String> typeFqnParts) {
      return PythonType.UNKNOWN;
    }
  }
}