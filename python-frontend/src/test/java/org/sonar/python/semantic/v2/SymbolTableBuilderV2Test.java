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
package org.sonar.python.semantic.v2;

import java.util.List;
import java.util.Set;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAliasStatement;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.tree.TreeUtils;

class SymbolTableBuilderV2Test {


  @Test
  void testSymbolTableModuleSymbols() {
    FileInput fileInput = PythonTestUtils.parse(
      """
        import lib
        
        v = lib.foo()
        a = lib.A()
        b = a.do_something()
        x = 42
        
        def script_do_something(param):
            c = 42
            return c
        
        script_do_something()
        """
    );

    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();

    var moduleSymbols = symbolTable.getSymbolsByRootTree(fileInput);
    Assertions.assertThat(moduleSymbols)
      .hasSize(6)
      .extracting(SymbolV2::name)
      .contains("lib", "a", "b", "v", "x", "script_do_something");
  }

  @Test
  void testSymbolTableLocalSymbols() {
    FileInput fileInput = PythonTestUtils.parse(
      """
        import lib
        a = lib.A()
        def script_do_something(param):
            c = 42
            return c
        """
    );

    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();

    var localSymbols = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance)
      .map(symbolTable::getSymbolsByRootTree)
      .orElseGet(Set::of);

    Assertions.assertThat(localSymbols)
      .hasSize(2)
      .extracting(SymbolV2::name)
      .contains("param", "c");
  }

  @Test
  void testSymbolTableGlobalSymbols() {
    FileInput fileInput = PythonTestUtils.parse(
      """
        global a
        def script_do_something(param):
            global b
            b = 42
        
        """
    );

    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();

    var localSymbols = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance)
      .map(symbolTable::getSymbolsByRootTree)
      .orElseGet(Set::of);
    var moduleSymbols = symbolTable.getSymbolsByRootTree(fileInput);

    Assertions.assertThat(localSymbols)
      .hasSize(1)
      .extracting(SymbolV2::name)
      .contains("param");

    Assertions.assertThat(moduleSymbols)
      .hasSize(3)
      .extracting(SymbolV2::name)
      .contains("a", "b", "script_do_something");
  }

  @Test
  void testNonLocalSymbol() {
    FileInput fileInput = PythonTestUtils.parse(
      """
        def foo():
          a = 42
          def inner(param):
            nonlocal a
            a = "hello"
            print(a)
        
        """
    );

    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();

    var moduleSymbols = symbolTable.getSymbolsByRootTree(fileInput);
    List<FunctionDef> functionDefs = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.FUNCDEF));

    Set<SymbolV2> fooSymbols = symbolTable.getSymbolsByRootTree(functionDefs.get(0));
    Set<SymbolV2> innerSymbols = symbolTable.getSymbolsByRootTree(functionDefs.get(1));

    Assertions.assertThat(moduleSymbols).extracting(SymbolV2::name).containsExactlyInAnyOrder("foo");
    Assertions.assertThat(fooSymbols).extracting(SymbolV2::name).containsExactlyInAnyOrder("a", "inner");
    Assertions.assertThat(innerSymbols).extracting(SymbolV2::name).containsExactlyInAnyOrder("param");
  }

  @Test
  void testNameSymbols() {
    FileInput fileInput = PythonTestUtils.parse(
      """
        import lib
        
        v = lib.foo()
        a = lib.A()
        b = a.do_something()
        x = 42
        
        def script_do_something(param):
            return 42
        
        script_do_something()
        """
    );

    new SymbolTableBuilderV2(fileInput)
      .build();

    var statements = fileInput.statements().statements();

    {
      var importStatement = (ImportName) statements.get(0);
      var libName = importStatement.modules().get(0).dottedName().names().get(0);
      assertNameSymbol(libName, "lib", 3);
    }

    {
      var assignmentStatement = (AssignmentStatement) statements.get(1);
      var vName = (Name) assignmentStatement.children().get(0).children().get(0);
      assertNameSymbol(vName, "v", 1);

      var libName = (Name) assignmentStatement.children().get(2).children().get(0).children().get(0);
      assertNameSymbol(libName, "lib", 3);
    }

    {
      var assignmentStatement = (AssignmentStatement) statements.get(2);
      var aName = (Name) assignmentStatement.children().get(0).children().get(0);
      assertNameSymbol(aName, "a", 2);

      var libName = (Name) assignmentStatement.children().get(2).children().get(0).children().get(0);
      assertNameSymbol(libName, "lib", 3);
    }

    {
      var assignmentStatement = (AssignmentStatement) statements.get(3);
      var bName = (Name) assignmentStatement.children().get(0).children().get(0);
      assertNameSymbol(bName, "b", 1);

      var aName = (Name) assignmentStatement.children().get(2).children().get(0).children().get(0);
      assertNameSymbol(aName, "a", 2);
    }

    {
      var assignmentStatement = (AssignmentStatement) statements.get(4);
      var xName = (Name) assignmentStatement.children().get(0).children().get(0);
      assertNameSymbol(xName, "x", 1);
    }

    {
      var functionDef = (FunctionDef) statements.get(5);
      var scriptFunctionName = (Name) functionDef.name();
      assertNameSymbol(scriptFunctionName, "script_do_something", 2);
    }

    {
      var functionCallName = (Name) statements.get(6).children().get(0).children().get(0);
      assertNameSymbol(functionCallName, "script_do_something", 2);
    }

  }


  @Test
  @Disabled("SONARPY-2259 primitives do not have a symbol")
  void primitives_have_symbol() {
    FileInput fileInput = PythonTestUtils.parse("""
      x = 3
      x is int
      
      float(x)
      """);

    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();

    var moduleSymbols = symbolTable.getSymbolsByRootTree(fileInput);
    Assertions.assertThat(moduleSymbols)
      .extracting(SymbolV2::name)
      .containsExactlyInAnyOrder("x", "int", "float");
  }

  @Test
  @Disabled("SONARPY-2260 variables which are never written to do not have a symbol")
  void never_written_variables_have_symbol() {
    FileInput fileInput = PythonTestUtils.parse("""
      x + 3
      if x == 3: pass
      """);

    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();

    var moduleSymbols = symbolTable.getSymbolsByRootTree(fileInput);
    Assertions.assertThat(moduleSymbols)
      .extracting(SymbolV2::name)
      .containsExactlyInAnyOrder("x");
  }


  @Test
  void class_type_param_bound_is_resolved_in_enclosing_scope() {
    FileInput fileInput = PythonTestUtils.parse("""
        class BaseBound:
            pass
        class Foo[T: BaseBound]:
            pass
        """);
    new SymbolTableBuilderV2(fileInput).build();
    ClassDef fooClass = (ClassDef) fileInput.statements().statements().get(1);
    Name boundName = (Name) fooClass.typeParams().typeParamsList().get(0).typeAnnotation().expression();
    Assertions.assertThat(boundName.symbolV2()).isNotNull();
    Assertions.assertThat(boundName.symbolV2().name()).isEqualTo("BaseBound");
  }

  @Test
  void type_alias_type_params_are_registered() {
    FileInput fileInput = PythonTestUtils.parse("type Alias[T] = list[T]");
    new SymbolTableBuilderV2(fileInput).build();
    TypeAliasStatement typeAlias = (TypeAliasStatement) fileInput.statements().statements().get(0);
    Name typeParamName = typeAlias.typeParams().typeParamsList().get(0).name();
    Assertions.assertThat(typeParamName.symbolV2()).isNotNull();
    Assertions.assertThat(typeParamName.symbolV2().usages())
      .extracting(UsageV2::kind)
      .contains(UsageV2.Kind.TYPE_PARAM_DECLARATION);
  }

  @Test
  void walrusInComprehensionBindsToEnclosingScope() {
    FileInput fileInput = PythonTestUtils.parse("""
      values = [(data := []) for x in range(1)]
      data
      """);
    new SymbolTableBuilderV2(fileInput).build();

    List<Name> dataNames = PythonTestUtils.<Name>getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.NAME))
      .stream()
      .filter(name -> "data".equals(name.name()))
      .toList();

    Assertions.assertThat(dataNames).hasSize(2);
    Assertions.assertThat(dataNames.get(0).symbolV2()).isSameAs(dataNames.get(1).symbolV2());
    Assertions.assertThat(dataNames.get(0).symbolV2().usages())
      .filteredOn(usage -> usage.kind() == UsageV2.Kind.ASSIGNMENT_LHS)
      .hasSize(1);
  }

  private static void assertNameSymbol(Name name, String expectedSymbolName, int expectedUsagesCount) {
    var symbol = name.symbolV2();
    Assertions.assertThat(symbol).isNotNull();
    Assertions.assertThat(symbol.name()).isEqualTo(expectedSymbolName);
    Assertions.assertThat(symbol.usages()).hasSize(expectedUsagesCount);
  }
}
