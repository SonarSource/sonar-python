/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import java.util.Objects;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.tree.ClassDefImpl;

import javax.annotation.Nullable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.functionSymbol;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.semantic.SymbolUtils.pathOf;
import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

public class SymbolUtilsTest {

  @Test
  public void package_name_by_file() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/__init__.py"), baseDir)).isEqualTo("sound");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/__init__.py"), baseDir)).isEqualTo("sound.formats");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/wavread.py"), baseDir)).isEqualTo("sound.formats");
  }

  @Test
  public void fqn_by_package_with_subpackage() {
    assertThat(SymbolUtils.fullyQualifiedModuleName("", "foo.py")).isEqualTo("foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("foo", "__init__.py")).isEqualTo("foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("foo", "foo.py")).isEqualTo("foo.foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("foo", "foo")).isEqualTo("foo.foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("curses", "ascii.py")).isEqualTo("curses.ascii");
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


  @Test
  public void get_overridden_method() {
    FileInput file = PythonTestUtils.parse( new SymbolTableBuilder("my_package", pythonFile("my_module.py")),
      "def foo(): pass",
      "def foo2():",
      "  def foo3(): pass",
      "class A:",
      "  def foo4(): pass",
      "class B:",
      "  def foo5(): pass",
      "  foo_int: int",
      "class C(B):",
      "  def foo5(): pass",
      "  def foo6(): pass",
      "  def foo_int(): pass",
      "class D(object):",
      "  def foo7(): pass",
      "class E(foo2):",
      "  def foo8(): pass"
    );

    FunctionSymbol foo = (FunctionSymbol) descendantFunction(file, "foo").name().symbol();
    FunctionSymbol foo2 = (FunctionSymbol) descendantFunction(file, "foo2").name().symbol();
    FunctionSymbol foo3 = (FunctionSymbol) descendantFunction(file, "foo3").name().symbol();
    FunctionSymbol foo4 = (FunctionSymbol) descendantFunction(file, "foo4").name().symbol();
    FunctionSymbol foo5 = (FunctionSymbol) descendantFunction(file, "foo5").name().symbol();
    FunctionSymbol foo5_override = (FunctionSymbol) ((FunctionDef) ((ClassDefImpl)file.statements().statements().get(4)).body().statements().get(0)).name().symbol();
    FunctionSymbol foo6 = (FunctionSymbol) descendantFunction(file, "foo6").name().symbol();
    FunctionSymbol foo7 = (FunctionSymbol) descendantFunction(file, "foo7").name().symbol();
    FunctionSymbol foo8 = (FunctionSymbol) descendantFunction(file, "foo8").name().symbol();
    FunctionSymbol foo_int = (FunctionSymbol) descendantFunction(file, "foo_int").name().symbol();
    assertThat(SymbolUtils.getOverriddenMethod(foo)).isEmpty();
    assertThat(SymbolUtils.getOverriddenMethod(foo2)).isEmpty();
    assertThat(SymbolUtils.getOverriddenMethod(foo3)).isEmpty();
    assertThat(SymbolUtils.getOverriddenMethod(foo4)).isEmpty();
    assertThat(SymbolUtils.getOverriddenMethod(foo5)).isEmpty();
    assertThat(SymbolUtils.getOverriddenMethod(foo5_override).get()).isEqualTo(foo5);
    assertThat(SymbolUtils.getOverriddenMethod(foo6)).isEmpty();
    assertThat(SymbolUtils.getOverriddenMethod(foo7)).isEmpty();
    assertThat(SymbolUtils.getOverriddenMethod(foo8)).isEmpty();
    assertThat(SymbolUtils.getOverriddenMethod(foo_int)).isEmpty();
  }

  @Nullable
  private static FunctionDef descendantFunction(Tree tree, String name) {
    if (tree.is(Tree.Kind.FUNCDEF)) {
      FunctionDef functionDef = (FunctionDef) tree;
      if (functionDef.name().name().equals(name)) {
        return functionDef;
      }
    }
    return tree.children().stream()
      .map(child -> descendantFunction(child, name))
      .filter(Objects::nonNull)
      .findFirst().orElse(null);
  }
}
