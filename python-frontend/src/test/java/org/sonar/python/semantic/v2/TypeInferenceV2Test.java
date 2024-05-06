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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
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
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.LIST_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.STR_TYPE;

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
    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();
    new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()), pythonFile, symbolTable).inferTypes(fileInput);

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

  @Test
  void inferTypeForReassignedVariables() {
    var root = inferTypes("""
      a = 42
      print(a)
      a = "Bob"
      print(a)
      a = "Marley"
      print(a)
      """);

    var aName = TreeUtils.firstChild(root.statements().statements().get(0), Name.class::isInstance)
      .map(Name.class::cast)
      .get();

    Assertions.assertThat(aName)
      .isNotNull()
      .extracting(Name::symbolV2)
      .isNotNull();

    var aSymbol = aName.symbolV2();
    Assertions.assertThat(aSymbol.usages()).hasSize(6);

    var types = aSymbol.usages()
      .stream()
      .map(UsageV2::tree)
      .map(Name.class::cast)
      .sorted(Comparator.comparing(n -> n.firstToken().line()))
      .map(Expression::typeV2)
      .map(PythonType::unwrappedType)
      .toList();

    Assertions.assertThat(types).hasSize(6)
      .containsExactly(INT_TYPE, INT_TYPE, STR_TYPE, STR_TYPE, STR_TYPE, STR_TYPE);
  }

  @Test
  void inferTypeForConditionallyReassignedVariables() {
    var root = inferTypes("""
      a = 42
      if (b):
        a = "Bob"
      print(a)
      """);

    var aName = TreeUtils.firstChild(root.statements().statements().get(0), Name.class::isInstance)
      .map(Name.class::cast)
      .get();

    Assertions.assertThat(aName)
      .isNotNull()
      .extracting(Name::symbolV2)
      .isNotNull();

    var aSymbol = aName.symbolV2();
    Assertions.assertThat(aSymbol.usages()).hasSize(3);

    var types = aSymbol.usages()
      .stream()
      .map(UsageV2::tree)
      .map(Name.class::cast)
      .sorted(Comparator.comparing(n -> n.firstToken().line()))
      .map(Expression::typeV2)
      .toList();

    Assertions.assertThat(types).hasSize(3);

    Assertions.assertThat(types.get(0)).isInstanceOf(ObjectType.class)
      .extracting(ObjectType.class::cast)
      .extracting(ObjectType::type)
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("int");

    Assertions.assertThat(types.get(1)).isInstanceOf(ObjectType.class)
      .extracting(ObjectType.class::cast)
      .extracting(ObjectType::type)
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("str");

    Assertions.assertThat(types.get(2)).isInstanceOf(UnionType.class);
    var type3 = (UnionType) types.get(2);
    Assertions.assertThat(type3.candidates())
      .hasSize(2)
      .extracting(ObjectType.class::cast)
      .extracting(ObjectType::type)
      .extracting(ClassType.class::cast)
      .extracting(ClassType::name)
      .containsExactlyInAnyOrder("int", "str");
  }

  @Test
  @Disabled("Resulting type should not be tuple")
  void unpacking_assignment() {
    assertThat(lastExpression(
      """
      x, = 42,
      x
      """
    ).typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void unpacking_assignment_2() {
    assertThat(lastExpression(
      """
      x, y = 42, 43
      x
      """
    ).typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void multiple_lhs_expressions() {
    assertThat(lastExpression(
      """
      x = y = 42
      x
      """
    ).typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void compoundAssignmentStr() {
    assertThat(lastExpression("""
      a = 42
      a += 1
      a
      """).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void compoundAssignmentList() {
    assertThat(lastExpression("""
        a = []
        b = 'world'
        a += b
        a
        """).typeV2().unwrappedType()).isEqualTo(LIST_TYPE);
  }

  @Test
  void annotation_with_reassignment() {
    assertThat(lastExpression("""
        a = "foo"
        b: int = a
        b
        """).typeV2().unwrappedType()).isEqualTo(STR_TYPE);
  }

  @Test
  @Disabled("ObjectType[PythonType.UNKNOWN] should just be PythonType.UNKNOWN")
  void call_expression() {
    assertThat(lastExpression(
      "f()").typeV2()).isEqualTo(PythonType.UNKNOWN);

    assertThat(lastExpression("""
        def f(): pass
        f()
        """).typeV2()).isEqualTo(PythonType.UNKNOWN);

    assertThat(lastExpression(
      """
      class A: pass
      A()
      """).typeV2().displayName()).contains("A");
  }

  @Test
  void variable_outside_function() {
    assertThat(lastExpression("a = 42; a").typeV2().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  @Disabled("Flow insensitive type inference scope issue")
  void variable_outside_function_2() {
    assertThat(lastExpression(
      """
        a = 42
        def foo(): a
        """).typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  @Disabled("Flow insensitive type inference scope issue")
  void variable_outside_function_3() {
    assertThat(lastExpression(
      """
      def foo():
        a = 42
      a
      """).type()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void variable_outside_function_4() {
    assertThat(lastExpression(
      """
      a = 42
      def foo():
        a = 'hello'
      a
      """).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
  }

  private static FileInput inferTypes(String lines) {
    return inferTypes(lines, new HashMap<>());
  }

  private static FileInput inferTypes(String lines, Map<String, Set<Symbol>> globalSymbols) {
    FileInput root = parseWithoutSymbols(lines);

    var symbolTable = new SymbolTableBuilderV2(root)
      .build();
    new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.from(globalSymbols)), pythonFile, symbolTable).inferTypes(root);
    return root;
  }

  public static Expression lastExpression(String lines) {
    FileInput fileInput = inferTypes(lines);
    Statement statement = lastStatement(fileInput.statements());
    if (!(statement instanceof ExpressionStatement)) {
      assertThat(statement).isInstanceOf(FunctionDef.class);
      FunctionDef fnDef = (FunctionDef) statement;
      statement = lastStatement(fnDef.body());
    }
    assertThat(statement).isInstanceOf(ExpressionStatement.class);
    List<Expression> expressions = ((ExpressionStatement) statement).expressions();
    return expressions.get(expressions.size() - 1);
  }

  private static Statement lastStatement(StatementList statementList) {
    List<Statement> statements = statementList.statements();
    return statements.get(statements.size() - 1);
  }
}
