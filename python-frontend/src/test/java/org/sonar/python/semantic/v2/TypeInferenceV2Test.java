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
package org.sonar.python.semantic.v2;

import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.PythonType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;

class TypeInferenceV2Test {
  private static FileInput fileInput;

  static PythonFile pythonFile = PythonTestUtils.pythonFile("");

  @BeforeAll
  static void init() {
    var context = TestPythonVisitorRunner.createContext(new File("src/test/resources/semantic/v2/script.py"));
    fileInput = context.rootTree();
  }

  @Test
  void test() {
    var pythonFile = PythonTestUtils.pythonFile("script.py");
    var builder = new SymbolTableBuilderV2();
    builder.visitFileInput(fileInput);
    var typeInferenceV2 = new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()), pythonFile);
    fileInput.accept(typeInferenceV2);

    System.out.println("hello");
  }

  @Test
  void testTypeshedImports() {
    FileInput root = inferTypes("""
      import cryptography.hazmat.primitives.asymmetric
      import datetime
      a = datetime.date(year=2023, month=12, day=1)
      """);

    var importName = (ImportName) root.statements().statements().get(0);
    var importedNames = importName.modules().get(0).dottedName().names();
    assertThat(importedNames.get(0))
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .extracting(PythonType::name)
      .isEqualTo("cryptography");
    assertThat(importedNames.get(1))
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .extracting(PythonType::name)
      .isEqualTo("hazmat");
    assertThat(importedNames.get(2))
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .extracting(PythonType::name)
      .isEqualTo("primitives");
    assertThat(importedNames.get(3))
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .extracting(PythonType::name)
      .isEqualTo("asymmetric");

    importName = (ImportName) root.statements().statements().get(1);
    importedNames = importName.modules().get(0).dottedName().names();
    assertThat(importedNames.get(0))
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .extracting(PythonType::name)
      .isEqualTo("datetime");
  }

  @Test
  void testUnresolvedImports() {
    FileInput root = inferTypes("""
      import something.unknown
      """);

    var importName = (ImportName) root.statements().statements().get(0);
    var importedNames = importName.modules().get(0).dottedName().names();
    assertThat(importedNames.get(0))
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .extracting(PythonType::name)
      .isEqualTo("something");
    assertThat(importedNames.get(1))
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .extracting(PythonType::name)
      .isEqualTo("unknown");
  }

  @Test
  void testProjectLevelSymbolTableImports() {
    var classSymbol = new ClassSymbolImpl("C", "something.known.C");

    var root = inferTypes("""
      import something.known
      """, Map.of("something", new HashSet<>(), "something.known", Set.of(classSymbol)));

    var importName = (ImportName) root.statements().statements().get(0);
    var importedNames = importName.modules().get(0).dottedName().names();
    assertThat(importedNames.get(0))
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .extracting(PythonType::name)
      .isEqualTo("something");
    assertThat(importedNames.get(1))
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .matches(type -> {
        assertThat(type.resolveMember("C").get())
          .isInstanceOf(ClassType.class);
        return true;
      })
      .extracting(PythonType::name)
      .isEqualTo("known");
  }

  @Test
  void testImportWithAlias() {
    FileInput root = inferTypes("""
      import datetime as d
      """);

    var importName = (ImportName) root.statements().statements().get(0);
    var aliasedName = importName.modules().get(0);

    var importedNames = aliasedName.dottedName().names();
    assertThat(importedNames)
      .flatExtracting(Expression::typeV2)
      .allMatch(PythonType.UNKNOWN::equals);
    
    assertThat(aliasedName.alias())
      .extracting(Expression::typeV2)
      .isInstanceOf(ModuleType.class)
      .extracting(PythonType::name)
      .isEqualTo("datetime");
  }

  @Test
  void testImportFrom() {
    FileInput root = inferTypes("""
      from datetime import date
      """);

    var importFrom = (ImportFrom) root.statements().statements().get(0);
    var type = importFrom.importedNames().get(0).dottedName().names().get(0).typeV2();

    Assertions.assertThat(type)
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("date");
  }

  @Test
  void testImportFromWithAlias() {
    FileInput root = inferTypes("""
      from datetime import date as d
      """);

    var importFrom = (ImportFrom) root.statements().statements().get(0);
    var type1 = importFrom.importedNames().get(0).dottedName().names().get(0).typeV2();
    var type2 = importFrom.importedNames().get(0).alias().typeV2();

    Assertions.assertThat(type1)
      .isEqualTo(PythonType.UNKNOWN);

    Assertions.assertThat(type2)
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("date");
  }

  @Test
  void simpleFunctionDef() {
    FileInput root = inferTypes("""
      def foo(a, b, c): ...
      foo(1,2,3)
      """);

    FunctionDef functionDef = (FunctionDef) root.statements().statements().get(0);

    FunctionType functionType = (FunctionType) functionDef.name().typeV2();
    assertThat(functionType.name()).isEqualTo("foo");
    assertThat(functionType.hasVariadicParameter()).isFalse();
    assertThat(functionType.parameters()).hasSize(3);

    CallExpression callExpression = ((CallExpression) TreeUtils.firstChild(root, t -> t.is(Tree.Kind.CALL_EXPR)).get());
    assertThat(callExpression.callee().typeV2()).isInstanceOf(FunctionType.class);
  }

  @Test
  void inferTypeForBuiltins() {
    FileInput root = inferTypes("""
      a = list
      """);

    var assignmentStatement = (AssignmentStatement) root.statements().statements().get(0);
    var assignedType = assignmentStatement.assignedValue().typeV2();

    assertThat(assignedType)
      .isNotNull()
      .isInstanceOf(ClassType.class);

    assertThat(assignedType.resolveMember("append"))
      .isPresent()
      .get()
      .isInstanceOf(FunctionType.class);
  }

  @Test
  void inferTypeForReassignedBuiltins() {
    FileInput root = inferTypes("""
      def foo():
        global list
        list = 42
        list = "hello"
        list
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var expressionStatement = (ExpressionStatement) functionDef.body().statements().get(3);
    Assertions.assertThat(expressionStatement.expressions().get(0).typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  private FileInput inferTypes(String lines) {
    return inferTypes(lines, new HashMap<>());
  }

  private FileInput inferTypes(String lines, Map<String, Set<Symbol>> globalSymbols) {
    FileInput root = parseWithoutSymbols(lines);
    var symbolTableBuilderV2 = new SymbolTableBuilderV2();
    root.accept(symbolTableBuilderV2);
    var typeInferenceV2 = new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.from(globalSymbols)), pythonFile);
    root.accept(typeInferenceV2);
    return root;
  }
}
