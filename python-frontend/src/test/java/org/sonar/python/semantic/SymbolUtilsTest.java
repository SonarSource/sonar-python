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

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.tree.ClassDefImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.functionSymbol;
import static org.sonar.python.PythonTestUtils.getLastDescendant;
import static org.sonar.python.PythonTestUtils.lastFunctionSymbolWithName;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.semantic.SymbolUtils.getFirstAlternativeIfEqualArgumentNames;
import static org.sonar.python.semantic.SymbolUtils.isEqualParameterCountAndNames;
import static org.sonar.python.semantic.SymbolUtils.pathOf;
import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

class SymbolUtilsTest {

  @Test
  void package_name_by_file() {
    String baseDir = new File("src/test/resources").getAbsoluteFile().getAbsolutePath();
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/__init__.py"), baseDir)).isEqualTo("sound");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/__init__.py"), baseDir)).isEqualTo("sound.formats");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/wavread.py"), baseDir)).isEqualTo("sound.formats");
  }

  @Test
  void fqn_by_package_with_subpackage() {
    assertThat(SymbolUtils.fullyQualifiedModuleName("", "foo.py")).isEqualTo("foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("foo", "__init__.py")).isEqualTo("foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("foo", "foo.py")).isEqualTo("foo.foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("foo", "foo")).isEqualTo("foo.foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("curses", "ascii.py")).isEqualTo("curses.ascii");
  }

  @Test
  void path_of() throws IOException, URISyntaxException {
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
  void first_parameter_offset() {
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

    functionSymbol = functionSymbol("class A:\n  @abstractmethod\n  def method(self, b): pass");
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, true)).isZero();
    assertThat(SymbolUtils.firstParameterOffset(functionSymbol, false)).isEqualTo(1);

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
  void get_overridden_method() {
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
      "  def foo8(): pass",
      "class MyStr(str):",
      "  def capitalize(self): pass"
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
    FunctionSymbol capitalize = (FunctionSymbol) descendantFunction(file, "capitalize").name().symbol();
    assertThat(SymbolUtils.getOverriddenMethod(foo)).isEmpty();
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo)).isFalse();
    assertThat(SymbolUtils.getOverriddenMethod(foo2)).isEmpty();
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo2)).isFalse();
    assertThat(SymbolUtils.getOverriddenMethod(foo3)).isEmpty();
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo3)).isFalse();
    assertThat(SymbolUtils.getOverriddenMethod(foo4)).isEmpty();
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo4)).isFalse();
    assertThat(SymbolUtils.getOverriddenMethod(foo5)).isEmpty();
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo5)).isFalse();
    assertThat(SymbolUtils.getOverriddenMethod(foo5_override).get()).isEqualTo(foo5);
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo5_override)).isTrue();
    assertThat(SymbolUtils.getOverriddenMethod(foo6)).isEmpty();
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo6)).isFalse();
    assertThat(SymbolUtils.getOverriddenMethod(foo7)).isEmpty();
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo7)).isFalse();
    assertThat(SymbolUtils.getOverriddenMethod(foo8)).isEmpty();
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo8)).isTrue();
    assertThat(SymbolUtils.getOverriddenMethod(foo_int)).isEmpty();
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo_int)).isTrue();
    assertThat(SymbolUtils.getOverriddenMethod(capitalize, SymbolUtils::getFirstAlternativeIfEqualArgumentNames)).isNotEmpty();

    assertThat(SymbolUtils.canBeAnOverridingMethod(null)).isTrue();
    String[] strings = {
      "class F:",
      "  def foo9(): pass",
      "class F:",
      "  def foo9(): pass",
      "  def bar9(): pass",
      "class G(F):",
      "  def foo9(): pass",
      "  def bar9(): pass"
    };
    FunctionSymbol bar9 = lastFunctionSymbolWithName("bar9", strings);
    assertThat(SymbolUtils.canBeAnOverridingMethod(bar9)).isTrue();

    FunctionSymbol foo9 = lastFunctionSymbolWithName("foo9", strings);
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo9)).isTrue();

    // coverage

    FunctionDef functionDef = getLastDescendant(parse("def foo10(): ..."), t -> t.is(Tree.Kind.FUNCDEF));
    FunctionSymbolImpl foo10 = new FunctionSymbolImpl(functionDef, "mod.foo", pythonFile("mod.py"));
    foo10.setOwner(new SymbolImpl("some", "some"));
    assertThat(SymbolUtils.canBeAnOverridingMethod(foo10)).isFalse();
  }

  @Test
  void getFunctionSymbolsTest() {
    assertThat(SymbolUtils.getFunctionSymbols(null)).isEmpty();

    var file = PythonTestUtils.parse( new SymbolTableBuilder("my_package", pythonFile("my_module.py")),
      "class MyStr(str):",
      "  def capitalize(self): pass"
    );
    var capitalize = (FunctionSymbolImpl) descendantFunction(file, "capitalize").name().symbol();
    assertThat(capitalize).isNotNull();
    assertThat(SymbolUtils.getFunctionSymbols(capitalize)).isNotEmpty().contains(capitalize);


    var owner = (ClassSymbol) capitalize.owner();
    assertThat(SymbolUtils.getFunctionSymbols(owner)).isEmpty();
    var capitalizeParentSymbol = ((ClassSymbol) owner.superClasses().get(0)).resolveMember("capitalize").orElse(null);
    assertThat(SymbolUtils.getFunctionSymbols(capitalizeParentSymbol)).isNotEmpty();
  }

  @Test
  void getFirstAlternativeIfEqualArgumentNamesTest() {
    var file = PythonTestUtils.parse( new SymbolTableBuilder("my_package", pythonFile("my_module.py")),
      """
        class MyClass(dict):
          def get(self, key): ...
        
        def foo1(a, b, c): ...
        def foo2(a, b, c): ...
        def bar(b, c, a): ...
        def qix(c, a, b): ...
        """);
    FunctionSymbol getMethod = (FunctionSymbol) descendantFunction(file, "get").name().symbol();
    List<FunctionSymbol> overriddenMethods = SymbolUtils.getOverriddenMethods(getMethod);
    assertThat(isEqualParameterCountAndNames(overriddenMethods)).isFalse();
    assertThat(getFirstAlternativeIfEqualArgumentNames(overriddenMethods)).isEmpty();

    FunctionSymbol foo1 = (FunctionSymbol) descendantFunction(file, "foo1").name().symbol();
    FunctionSymbol foo2 = (FunctionSymbol) descendantFunction(file, "foo2").name().symbol();
    FunctionSymbol bar = (FunctionSymbol) descendantFunction(file, "bar").name().symbol();
    FunctionSymbol qix = (FunctionSymbol) descendantFunction(file, "qix").name().symbol();
    assertThat(isEqualParameterCountAndNames(List.of(foo1, bar, qix))).isFalse();
    assertThat(isEqualParameterCountAndNames(List.of(foo1, foo2))).isTrue();
    assertThat(getFirstAlternativeIfEqualArgumentNames(List.of(foo1, foo2))).isPresent();
  }

  @Test
  void isEqualParameterCountAndNamesTest() {
    var file = PythonTestUtils.parse( new SymbolTableBuilder("my_package", pythonFile("my_module.py")),
      "class A:",
      "  def foo1(self, a):",
      "    ...",
      "class B:",
      "  def foo2(self, a):",
      "    ...",
      "class C:",
      "  def foo3(self, b):",
      "    ..."
    );

    FunctionSymbol foo1 = (FunctionSymbol) descendantFunction(file, "foo1").name().symbol();
    FunctionSymbol foo2 = (FunctionSymbol) descendantFunction(file, "foo2").name().symbol();
    FunctionSymbol foo3 = (FunctionSymbol) descendantFunction(file, "foo3").name().symbol();

    assertThat(foo1).isNotNull();
    assertThat(foo2).isNotNull();
    assertThat(foo3).isNotNull();
    assertThat(SymbolUtils.isEqualParameterCountAndNames(List.of(foo1, foo2))).isTrue();
    assertThat(SymbolUtils.isEqualParameterCountAndNames(List.of(foo1, foo3))).isFalse();
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
