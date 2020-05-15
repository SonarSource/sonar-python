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

import com.google.common.base.Functions;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FileInput;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.functionSymbol;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.semantic.SymbolUtils.pathOf;
import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

public class SymbolUtilsTest {

  @Test
  public void global_symbols() {
    FileInput tree = parseWithoutSymbols(
      "obj1 = 42",
      "obj2: int = 42",
      "def fn(): pass",
      "class A: pass"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "", pythonFile("mod.py"));
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("obj1", "obj2", "fn", "A");
    assertThat(globalSymbols).extracting(Symbol::fullyQualifiedName).containsExactlyInAnyOrder("mod.obj1", "mod.obj2", "mod.fn", "mod.A");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  public void global_symbols_private_by_convention() {
    // although being private by convention, it's considered as exported
    FileInput tree = parseWithoutSymbols(
      "def _private_fn(): pass"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "mod", pythonFile("mod.py"));
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("_private_fn");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  public void local_symbols_not_exported() {
    FileInput tree = parseWithoutSymbols(
      "def fn():",
      "  def inner(): pass",
      "  class Inner_class: pass",
      "class A:",
      "  def meth(): pass"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "mod", pythonFile("mod.py"));
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("fn", "A");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  public void redefined_symbols() {
    FileInput tree = parseWithoutSymbols(
      "def fn(): pass",
      "def fn(): ...",
      "if True:",
      "  conditionally_defined = 1",
      "else:",
      "  conditionally_defined = 2"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "mod", pythonFile("mod.py"));
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("fn", "conditionally_defined");
    assertThat(globalSymbols).extracting(Symbol::usages).allSatisfy(usages -> assertThat(usages).isEmpty());
  }

  @Test
  public void function_symbols() {
    FileInput tree = parseWithoutSymbols(
      "def fn(): pass"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "mod", pythonFile("mod.py"));
    assertThat(globalSymbols).extracting(Symbol::kind).containsExactly(Symbol.Kind.FUNCTION);

    tree = parseWithoutSymbols(
      "def fn(): pass",
      "fn = 42"
    );
    globalSymbols = SymbolUtils.globalSymbols(tree, "mod", pythonFile("mod.py"));
    assertThat(globalSymbols).extracting(Symbol::kind).containsExactly(Symbol.Kind.OTHER);
  }

  @Test
  public void redefined_class_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "C = \"hello\"",
      "class C: ",
      "  pass");
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(fileInput, "mod", pythonFile("mod.py"));
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("C");
    assertThat(globalSymbols).extracting(Symbol::kind).allSatisfy(k -> assertThat(Symbol.Kind.CLASS.equals(k)).isFalse());
  }

  @Test
  public void classdef_with_missing_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "class C: ",
      "  pass",
      "global C");

    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(fileInput, "mod", pythonFile("mod.py"));
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("C");
    // TODO: Global statements should not alter the kind of a symbol
    assertThat(globalSymbols).extracting(Symbol::kind).allSatisfy(k -> assertThat(Symbol.Kind.OTHER.equals(k)).isTrue());
  }

  @Test
  public void class_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "class C: ",
      "  pass");
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(fileInput, "mod", pythonFile("mod.py"));
    assertThat(globalSymbols).hasSize(1);
    Symbol cSymbol = globalSymbols.iterator().next();
    assertThat(cSymbol.name()).isEqualTo("C");
    assertThat(cSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) cSymbol).superClasses()).isEmpty();

    fileInput = parseWithoutSymbols(
      "class A: pass",
      "class C(A): ",
      "  pass");
    globalSymbols = SymbolUtils.globalSymbols(fileInput, "mod", pythonFile("mod.py"));
    Map<String, Symbol> symbols = globalSymbols.stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    cSymbol = symbols.get("C");
    assertThat(cSymbol.name()).isEqualTo("C");
    assertThat(cSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) cSymbol).superClasses()).hasSize(1);

    // for the time being, we only consider symbols defined in the global scope
    fileInput = parseWithoutSymbols(
      "class A:",
      "  class A1: pass",
      "class C(A.A1): ",
      "  pass");
    globalSymbols = SymbolUtils.globalSymbols(fileInput, "mod", pythonFile("mod.py"));
    symbols = globalSymbols.stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    cSymbol = symbols.get("C");
    assertThat(cSymbol.name()).isEqualTo("C");
    assertThat(cSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) cSymbol).superClasses()).hasSize(1);
  }

  @Test
  public void class_inheriting_from_imported_symbol() {
    FileInput fileInput = parseWithoutSymbols(
      "from mod import A",
      "import mod2",
      "class C(A): ",
      "  pass",
      "class D(mod2.B):",
      "  pass");

    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(fileInput, "mod", pythonFile("mod.py"));
    Map<String, Symbol> symbols = globalSymbols.stream().collect(Collectors.toMap(Symbol::name, Functions.identity()));
    Symbol cSymbol = symbols.get("C");
    assertThat(cSymbol.name()).isEqualTo("C");
    assertThat(cSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) cSymbol).superClasses()).hasSize(1);
    assertThat(((ClassSymbol) cSymbol).superClasses().get(0).fullyQualifiedName()).isEqualTo("mod.A");
    Symbol dSymbol = symbols.get("D");
    assertThat(dSymbol.name()).isEqualTo("D");
    assertThat(dSymbol.kind()).isEqualTo(Symbol.Kind.CLASS);
    assertThat(((ClassSymbol) dSymbol).superClasses()).hasSize(1);
    assertThat(((ClassSymbol) dSymbol).superClasses().get(0).fullyQualifiedName()).isEqualTo("mod2.B");
  }

  @Test
  public void symbol_duplicated_by_wildcard_import() {
    FileInput tree = parseWithoutSymbols(
      "def nlargest(n, iterable): ...",
      "from _heapq import *",
      "def nlargest(n, iterable, key=None): ..."
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "", pythonFile("mod.py"));
    assertThat(globalSymbols).isEmpty();

    tree = parseWithoutSymbols(
      "nonlocal nlargest",
      "from _heapq import *",
      "def nlargest(n, iterable, key=None): ..."
    );
    globalSymbols = SymbolUtils.globalSymbols(tree, "", pythonFile("mod.py"));
    assertThat(globalSymbols).isEmpty();
  }

  @Test
  public void package_name_by_file() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/__init__.py"), baseDir)).isEqualTo("sound");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/__init__.py"), baseDir)).isEqualTo("sound.formats");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/wavread.py"), baseDir)).isEqualTo("sound.formats");
  }

  @Test
  public void class_having_itself_as_superclass_should_not_trigger_error() {
    FileInput fileInput = parseWithoutSymbols("class A(A): pass");
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(fileInput, "mod", pythonFile("mod.py"));
    ClassSymbol a = (ClassSymbol) globalSymbols.iterator().next();
    assertThat(a.superClasses()).containsExactly(a);
  }

  @Test
  public void global_symbols_stdlib_imports() {
    FileInput tree = parseWithoutSymbols(
      "from time import time",
      "from threading import Thread",
      "from datetime import TimezoneMixin as tz",
      "import unknown",
      "from mod import *"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "", pythonFile("mod.py"));
    assertThat(globalSymbols).isEmpty();
  }

  @Test
  public void module_importing_itself() {
    FileInput tree = parseWithoutSymbols(
      "from mod import *",
      "from mod import smth"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "", pythonFile("mod.py"));
    assertThat(globalSymbols).isEmpty();
  }

  @Test
  public void path_of() throws IOException, URISyntaxException {
    PythonFile pythonFile = Mockito.mock(PythonFile.class);
    URI uri = Files.createTempFile("foo.py", "py").toUri();
    Mockito.when(pythonFile.uri()).thenReturn(uri);
    assertThat(pathOf(pythonFile)).isEqualTo(Paths.get(uri));

    uri = new URI("myscheme", null, "/file1.py", null);

    Mockito.when(pythonFile.uri()).thenReturn(uri);
    assertThat(pathOf(pythonFile)).isNull();

    Mockito.when(pythonFile.uri()).thenThrow(InvalidPathException.class);
    assertThat(pathOf(pythonFile)).isNull();
  }

  @Test
  public void first_parameter_offset() {
    FunctionSymbol functionSymbol = functionSymbol("class A:\n  def method(self, *args): pass");
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, true)).isEqualTo(0);
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, false)).isEqualTo(1);

    functionSymbol = functionSymbol("class A:\n  @staticmethod\n  def method(a, b): pass");
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, true)).isEqualTo(0);
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, false)).isEqualTo(0);

    functionSymbol = functionSymbol("class A:\n  @classmethod\n  def method(a, b): pass");
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, true)).isEqualTo(1);
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, false)).isEqualTo(1);

    functionSymbol = functionSymbol("class A:\n  @staticmethod\n  @another_decorator\n  def method(a, b): pass");
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, true)).isEqualTo(-1);
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, false)).isEqualTo(-1);

    functionSymbol = functionSymbol("class A:\n  @unknown_decorator\n  def method(self, *args): pass");
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, false)).isEqualTo(-1);
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, true)).isEqualTo(-1);

    functionSymbol = functionSymbol("class A:\n  def method((a, b), c): pass");
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, false)).isEqualTo(-1);
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, true)).isEqualTo(-1);

    functionSymbol = functionSymbol("def function(): pass");
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, true)).isEqualTo(0);
  }
}
