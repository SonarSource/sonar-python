/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.plugins.python.api.types;

import org.junit.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.semantic.SymbolTableBuilder;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.semantic.ProjectLevelSymbolTable.from;

public class TypeShedTypesApiTest {
  @Test
  public void import_package() {
    parse(
      "import os"
    );
    assertThat(TypeShedTypesApi.getTypeShedSymbols().get("os")).isNotEmpty();
    // Transitive types will appear
    assertThat(TypeShedTypesApi.getTypeShedSymbols().get("builtins")).isNotEmpty();

    parse(
      "import io"
    );
    assertThat(TypeShedTypesApi.getTypeShedSymbols().get("io")).isNotEmpty();
    // Old imports not being erased
    assertThat(TypeShedTypesApi.getTypeShedSymbols().get("os")).isNotEmpty();
  }

  @Test
  public void import_non_existing_package() {
    parse(
      "import non_existing_package"
    );
    assertThat(TypeShedTypesApi.getTypeShedSymbols().get("non_existing_package")).isEmpty();
  }

  @Test
  public void project_symbols_not_included() {
    SymbolImpl exportedA = new SymbolImpl("a", "type_module.a");
    Map<String, Set<Symbol>> globalSymbols = Collections.singletonMap("type_module", new HashSet<>(Collections.singletonList(exportedA)));
    parse(
      new SymbolTableBuilder("type_package", pythonFile("type_module_2.py"), from(globalSymbols)),
      "from type_module import *",
      "print(a)"
    );

    assertThat(TypeShedTypesApi.getTypeShedSymbols()).doesNotContainKeys("type_module", "type_module_2", "type_package");
    assertThat(TypeShedTypesApi.getBuiltinGlobalSymbols()).doesNotContainKeys("type_module", "type_module_2", "type_package");
  }

  @Test
  public void built_in_types_imported_automatically() {
    parse(
      "a = 1"
    );
    assertThat(TypeShedTypesApi.getBuiltinGlobalSymbols().get("")).filteredOn(s -> "int".equals(s.name())).isNotEmpty();
  }
}
