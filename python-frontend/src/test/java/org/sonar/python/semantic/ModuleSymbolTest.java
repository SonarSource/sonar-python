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
package org.sonar.python.semantic;

import org.junit.Test;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.ModuleSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.tree.TreeUtils;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.pythonFile;

public class ModuleSymbolTest {
  @Test
  public void symbol_functionality() {
    FunctionSymbol functionSymbol = functionSymbol("def fn(): pass");
    ModuleSymbol moduleSymbol = moduleSymbol("mod");
    assertThat(moduleSymbol.declaredMembers()).isEmpty();
    ((ModuleSymbolImpl) moduleSymbol).addMember(functionSymbol);
    assertThat(moduleSymbol.declaredMembers().iterator().next()).isEqualTo(functionSymbol);

    moduleSymbol = moduleSymbol("mod");
    ((ModuleSymbolImpl) moduleSymbol).addMembers(Collections.singletonList(functionSymbol));
    assertThat(moduleSymbol.declaredMembers().iterator().next()).isEqualTo(functionSymbol);
  }

  @Test
  public void module_import() {
    // Simple import creates module symbol
    FileInput tree = parse(
      "import pymysql");
    ImportName moduleImport = (ImportName) tree.statements().statements().get(0);
    ModuleSymbolImpl symbol = (ModuleSymbolImpl) moduleImport.modules().get(0).dottedName().names().get(0).symbol();
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.MODULE);
    // Method defined in TypeShed should exist
    Symbol functionSymbol = symbol.declaredMembers().stream().filter(s -> "thread_safe".equals(s.name())).findAny().get();
    assertThat(functionSymbol.is(Symbol.Kind.FUNCTION)).isTrue();
    // Thing that was not defined in TypeShed - should not exist
    assertThat(symbol.declaredMembers().stream().noneMatch(s -> "connections".equals(s.name()))).isTrue();

    // Non-existing symbol being added to module declaration
    tree = parse(
      "import pymysql",
      "pymysql.connections.Connection()");
    moduleImport = (ImportName) tree.statements().statements().get(0);
    symbol = (ModuleSymbolImpl) moduleImport.modules().get(0).dottedName().names().get(0).symbol();
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.MODULE);
    // But when it's referenced in the code - it is added to the module symbol
    assertThat(symbol.declaredMembers().stream().anyMatch(s -> "connections".equals(s.name()) && "pymysql.connections".equals(s.fullyQualifiedName()))).isTrue();
  }

  @Test
  public void imported_module_from_project_level_symbol_table() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.from(SymbolUtils.externalModulesSymbols());
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", pythonFile("my_module.py"), projectLevelSymbolTable),
      "from OpenSSL import SSL"
    );
    Symbol importedSslSymbol = tree.globalVariables().iterator().next();
    assertThat(importedSslSymbol.name()).isEqualTo("SSL");
    assertThat(importedSslSymbol.kind()).isEqualTo(Symbol.Kind.MODULE);
  }

  @Test
  public void copy_without_usages() {
    ModuleSymbolImpl foo = new ModuleSymbolImpl("module", "mod1");
    ModuleSymbolImpl otherFoo = new ModuleSymbolImpl("module", "mod2");
    AmbiguousSymbol ambiguousSymbol = AmbiguousSymbolImpl.create(new HashSet<>(Arrays.asList(foo, otherFoo)));
    AmbiguousSymbolImpl copy = ((AmbiguousSymbolImpl) ambiguousSymbol).copyWithoutUsages();
    assertThat(copy.is(Symbol.Kind.AMBIGUOUS)).isTrue();
    assertThat(copy.usages()).isEmpty();
    assertThat(copy).isNotEqualTo(ambiguousSymbol);
    assertThat(copy.alternatives()).doesNotContain(foo, otherFoo);
    assertThat(copy.alternatives()).extracting(Symbol::name, Symbol::fullyQualifiedName).containsExactlyInAnyOrder(tuple("module", "mod1"), tuple("module", "mod2"));
  }

  private FunctionSymbol functionSymbol(PythonFile pythonFile, String... code) {
    FileInput fileInput = parse(new SymbolTableBuilder(pythonFile), code);
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    ;
    return TreeUtils.getFunctionSymbolFromDef(functionDef);
  }

  private FunctionSymbol functionSymbol(String... code) {
    return functionSymbol(PythonTestUtils.pythonFile("foo"), code);
  }

  private ModuleSymbol moduleSymbol(String name) {
    return new ModuleSymbolImpl(name, name);
  }
}
