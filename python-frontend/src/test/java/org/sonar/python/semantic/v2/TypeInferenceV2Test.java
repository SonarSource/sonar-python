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

import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeSource;
import org.sonar.python.types.v2.UnionType;
import org.sonar.python.types.v2.UnknownType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.assertThatThrownBy;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.types.v2.TypesTestUtils.DICT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.LIST_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.NONE_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.SET_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.STR_TYPE;

class TypeInferenceV2Test {

  static PythonFile pythonFile = PythonTestUtils.pythonFile("");

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

    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.from(
      Map.of("something", new HashSet<>(), "something.known", Set.of(classSymbol)))
    );
    var root = inferTypes("""
      import something.known
      """, projectLevelTypeTable);

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
  void multipleFunctionDefinitions() {
    FileInput root = inferTypes("""
      def foo(a, b, c): ...
      foo(1, 2, 3)
      def foo(a, b): ...
      foo(1, 2)
      """);
    FunctionType firstFunctionType = (FunctionType) ((FunctionDef) root.statements().statements().get(0)).name().typeV2();
    FunctionType secondFunctionType = (FunctionType) ((FunctionDef) root.statements().statements().get(2)).name().typeV2();

    List<CallExpression> calls = PythonTestUtils.getAllDescendant(root, tree -> tree.is(Tree.Kind.CALL_EXPR));
    CallExpression firstCall = calls.get(0);
    CallExpression secondCall = calls.get(1);

    assertThat(firstCall.callee().typeV2()).isEqualTo(firstFunctionType);
    assertThat(secondCall.callee().typeV2()).isEqualTo(secondFunctionType);
  }

  @Test
  @Disabled("ClassDef not in CFG prevents us from getting this to work")
  void multipleClassDefinitions() {
    FileInput root = inferTypes("""
      class MyClass(int): ...
      MyClass()
      class MyClass(str): ...
      MyClass()
      """);
    ClassType firstClassType = (ClassType) ((ClassDef) root.statements().statements().get(0)).name().typeV2();
    ClassType secondClassType = (ClassType) ((ClassDef) root.statements().statements().get(2)).name().typeV2();

    List<CallExpression> calls = PythonTestUtils.getAllDescendant(root, tree -> tree.is(Tree.Kind.CALL_EXPR));
    CallExpression firstCall = calls.get(0);
    CallExpression secondCall = calls.get(1);

    assertThat(firstCall.callee().typeV2()).isEqualTo(firstClassType);
    assertThat(secondCall.callee().typeV2()).isEqualTo(secondClassType);
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
  void inferTypesInsideFunction1() {
    FileInput root = inferTypes("""
      x = 42
      def foo():
        x
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(1);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction2() {
    FileInput root = inferTypes("""
      x = 42
      def foo():
        x
      x = "hello"
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(1);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction3() {
    FileInput root = inferTypes("""
      x = "hello"
      def foo():
        x = 42
        x
      x = "world"
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(1);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void inferTypesInsideFunction4() {
    FileInput root = inferTypes("""
      def foo():
        x = 42
      x
      """);

    var lastExpressionStatement = (ExpressionStatement) root.statements().statements().get(root.statements().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction5() {
    FileInput root = inferTypes("""
      def foo(param: int):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().typeSource()).isEqualTo(TypeSource.TYPE_HINT);
  }

  @Test
  void inferTypesInsideFunction6() {
    FileInput root = inferTypes("""
      def foo(param: int):
        param = "hello"
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(STR_TYPE);
  }

  @Test
  void inferTypesInsideFunction7() {
    FileInput root = inferTypes("""
      def foo(param):
        param = "hello"
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(STR_TYPE);
  }

  @Test
  void inferTypesInsideFunction8() {
    FileInput root = inferTypes("""
      def foo(param: int):
        x = param
        x
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().typeSource()).isEqualTo(TypeSource.TYPE_HINT);
  }

  @Test
  void inferTypesInsideFunction9() {
    FileInput root = inferTypes("""
      def foo(param: list[int]):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    var type = (ObjectType) lastExpressionStatement.expressions().get(0).typeV2();
    Assertions.assertThat(type.unwrappedType()).isEqualTo(LIST_TYPE);
    Assertions.assertThat(type.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
    Assertions.assertThat(type.attributes())
      .extracting(PythonType::unwrappedType)
      .containsOnly(INT_TYPE);
  }

  @Test
  void inferTypesInsideFunction10() {
    FileInput root = inferTypes("""
      def foo(param: something_unknown):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction11() {
    FileInput root = inferTypes("""
      def foo(param: something_unknown[int]):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction12() {
    FileInput root = inferTypes("""
      o = "123"
      def foo(param: o):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(1);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    Assertions.assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction13() {
    FileInput root = inferTypes("""
      def foo(param: int | str):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    var type = (UnionType) lastExpressionStatement.expressions().get(0).typeV2();
    
    Assertions.assertThat(type.candidates())
      .extracting(PythonType::unwrappedType)
      .containsOnly(INT_TYPE, STR_TYPE);
  }

  @Test
  void inferTypesInsideFunction14() {
    FileInput root = inferTypes("""
      def foo(param: int):
        param
        param = "hello"
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var firstExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(0);
    var firstType = firstExpressionStatement.expressions().get(0).typeV2();
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(2);
    var lastType = lastExpressionStatement.expressions().get(0).typeV2();

    Assertions.assertThat(firstType.unwrappedType()).isEqualTo(INT_TYPE);
    Assertions.assertThat(lastType.unwrappedType()).isEqualTo(STR_TYPE);
  }

  @Test
  void inferTypesInsideFunction15() {
    FileInput root = inferTypes("""
      def foo3(a: int = "123"):
        a
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(0);
    var lastType = lastExpressionStatement.expressions().get(0).typeV2();

    Assertions.assertThat(lastType.unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void inferTypeForReassignedBuiltinsInsideFunction() {
    FileInput root = inferTypes("""
      def foo():
        global x
        x = 42
        x = "hello"
        x
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var expressionStatement = (ExpressionStatement) functionDef.body().statements().get(3);
    assertThat(expressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void global_variable() {
    assertThat(lastExpression("""
        global a
        a = 42
        a
        """).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void global_variable_builtin() {
    assertThat(lastExpression("""
        global list
        list = 42
        list
        """).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void conditional_assignment() {
    PythonType type = lastExpression("""
      if p:
        x = 42
      else:
        x = 'str'
      x
      """).typeV2().unwrappedType();
    assertThat(type).isInstanceOf(UnionType.class);
    assertThat(((UnionType) type).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
  }

  @Test
  void conditional_assignment_in_function() {
    FileInput fileInput = inferTypes("""
      def foo():
        if p:
          x = 42
        else:
          x = 'str'
        x
      """);
    var functionDef = (FunctionDef) fileInput.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() -1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2()).isInstanceOf(UnionType.class);
    assertThat(((UnionType) lastExpressionStatement.expressions().get(0).typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
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
  void annotation_without_reassignment() {
    assertThat(lastExpression("""
        a: int
        a
        """).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
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
  void variable_outside_function_2() {
    assertThat(lastExpression(
      """
        a = 42
        def foo(): a
        """).typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void variable_outside_function_3() {
    assertThat(lastExpression(
      """
      def foo():
        a = 42
      a
      """).typeV2()).isEqualTo(PythonType.UNKNOWN);
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

  @Test
  void recursive_function() {
    FileInput fileInput = inferTypes("""
      def my_recursive_func(n):
        ...
        my_recursive_func(n-1)
      """);
    FunctionDef functionDef = ((FunctionDef) fileInput.statements().statements().get(0));
    FunctionType functionType = (FunctionType) functionDef.name().typeV2();
    CallExpression call = (CallExpression) PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR)).get(0);
    assertThat(call.callee().typeV2()).isEqualTo(functionType);
  }

  @Test
  void recursive_function_try_except() {
    FileInput fileInput = inferTypes("""
      def recurser(x):
          if x is False:
              return
          recurser(False)
          try:
              recurser(False)
          except:
              recurser = None
              recurser(False)
          recurser(False)
      """);
    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));

    PythonType calleeType1 = calls.get(0).callee().typeV2();
    PythonType calleeType2 = calls.get(1).callee().typeV2();
    PythonType calleeType3 = calls.get(2).callee().typeV2();
    PythonType calleeType4 = calls.get(3).callee().typeV2();

    assertThat(calleeType1.unwrappedType()).isEqualTo(NONE_TYPE);
    assertThat(calleeType2.unwrappedType()).isEqualTo(NONE_TYPE);
    assertThat(calleeType3.unwrappedType()).isEqualTo(NONE_TYPE);
    assertThat(calleeType4.unwrappedType()).isEqualTo(NONE_TYPE);
  }

  @Test
  void recursive_function_try_except_2() {
    FileInput fileInput = inferTypes("""
      def wrapper():
          def recurser(x):
              ...
              recurser(False)
          try:
              recurser(True)
          finally:
              recurser = None
              recurser(True)
          recurser(True)
      """);

    FunctionDef wrapperDef = ((FunctionDef) fileInput.statements().statements().get(0));
    FunctionDef recurserDef = (FunctionDef) wrapperDef.body().statements().get(0);
    FunctionType functionType = (FunctionType) recurserDef.name().typeV2();

    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    PythonType calleeType1 = calls.get(0).callee().typeV2();
    PythonType calleeType2 = calls.get(1).callee().typeV2();
    PythonType calleeType3 = calls.get(2).callee().typeV2();
    PythonType calleeType4 = calls.get(3).callee().typeV2();

    assertThat(((UnionType) calleeType1).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(functionType, NONE_TYPE);
    assertThat(((UnionType) calleeType2).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(functionType, NONE_TYPE);
    assertThat(((UnionType) calleeType3).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(functionType, NONE_TYPE);
    assertThat(((UnionType) calleeType4).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(functionType, NONE_TYPE);
  }

  @Test
  void function_names_in_try_except_still_have_function_type() {
    FileInput fileInput = inferTypes("""
      def wrapper():
          def recurser(x):
              ...
          try:
              ...
          finally:
              def recurser(y):
                ...
      """);

    FunctionDef wrapperDef = ((FunctionDef) fileInput.statements().statements().get(0));

    List<FunctionDef> functionDefs = PythonTestUtils.getAllDescendant(wrapperDef, tree -> tree.is(Tree.Kind.FUNCDEF));
    FunctionDef recurserDef1 = functionDefs.get(0);
    FunctionDef recurserDef2 = functionDefs.get(1);
    FunctionType functionType1 = (FunctionType) recurserDef1.name().typeV2();
    FunctionType functionType2 = (FunctionType) recurserDef2.name().typeV2();

    assertThat(functionType1.parameters()).extracting(ParameterV2::name).containsExactlyInAnyOrder("x");
    assertThat(functionType2.parameters()).extracting(ParameterV2::name).containsExactlyInAnyOrder("y");
  }

  @Test
  void reassigned_class_try_except() {
    FileInput fileInput = inferTypes("""
      class MyClass:
          ...
      MyClass()
      try:
          MyClass()
      finally:
          MyClass = None
          MyClass()
      MyClass()
      """);

    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    PythonType calleeType1 = calls.get(0).callee().typeV2();
    PythonType calleeType2 = calls.get(1).callee().typeV2();
    PythonType calleeType3 = calls.get(2).callee().typeV2();
    PythonType calleeType4 = calls.get(3).callee().typeV2();

    assertThat(calleeType1).isEqualTo(PythonType.UNKNOWN);
    assertThat(calleeType2).isEqualTo(PythonType.UNKNOWN);
    assertThat(calleeType3).isEqualTo(PythonType.UNKNOWN);
    assertThat(calleeType4).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void flow_insensitive_when_try_except() {
    FileInput fileInput = inferTypes("""
      try:
        if p:
          x = 42
          type(x)
        else:
          x = "foo"
          type(x)
      except:
        type(x)
      """);

    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument firstX = (RegularArgument) calls.get(0).arguments().get(0);
    RegularArgument secondX = (RegularArgument) calls.get(1).arguments().get(0);
    RegularArgument thirdX = (RegularArgument) calls.get(2).arguments().get(0);
    assertThat(((UnionType) firstX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) secondX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) thirdX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
  }

  @Test
  void nested_try_except() {
    FileInput fileInput = inferTypes("""
        def f(p):
          try:
            if p:
              x = 42
              type(x)
            else:
              x = "foo"
              type(x)
          except:
            type(x)
        def g(p):
          if p:
            y = 42
            type(y)
          else:
            y = "hello"
            type(y)
          type(y)
        if cond:
          z = 42
          type(z)
        else:
          z = "hello"
          type(z)
        type(z)
        """);
    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument firstX = (RegularArgument) calls.get(0).arguments().get(0);
    RegularArgument secondX = (RegularArgument) calls.get(1).arguments().get(0);
    RegularArgument thirdX = (RegularArgument) calls.get(2).arguments().get(0);
    assertThat(((UnionType) firstX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) secondX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) thirdX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);

    RegularArgument firstY = (RegularArgument) calls.get(3).arguments().get(0);
    RegularArgument secondY = (RegularArgument) calls.get(4).arguments().get(0);
    RegularArgument thirdY = (RegularArgument) calls.get(5).arguments().get(0);
    assertThat(firstY.expression().typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(secondY.expression().typeV2().unwrappedType()).isEqualTo(STR_TYPE);
    assertThat(((UnionType) thirdY.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);

    RegularArgument firstZ = (RegularArgument) calls.get(6).arguments().get(0);
    RegularArgument secondZ = (RegularArgument) calls.get(7).arguments().get(0);
    RegularArgument thirdZ = (RegularArgument) calls.get(8).arguments().get(0);
    assertThat(firstZ.expression().typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(secondZ.expression().typeV2().unwrappedType()).isEqualTo(STR_TYPE);
    assertThat(((UnionType) thirdZ.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
  }

  @Test
  void nested_try_except_2() {
    FileInput fileInput = inferTypes("""
        try:
          if p:
            x = 42
            type(x)
          else:
            x = "foo"
            type(x)
        except:
          type(x)
        def g(p):
          if p:
            y = 42
            type(y)
          else:
            y = "hello"
            type(y)
          type(y)
        if cond:
          z = 42
          type(z)
        else:
          z = "hello"
          type(z)
        type(z)
        """);
    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument firstX = (RegularArgument) calls.get(0).arguments().get(0);
    RegularArgument secondX = (RegularArgument) calls.get(1).arguments().get(0);
    RegularArgument thirdX = (RegularArgument) calls.get(2).arguments().get(0);
    assertThat(((UnionType) firstX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) secondX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) thirdX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);

    RegularArgument firstY = (RegularArgument) calls.get(3).arguments().get(0);
    RegularArgument secondY = (RegularArgument) calls.get(4).arguments().get(0);
    RegularArgument thirdY = (RegularArgument) calls.get(5).arguments().get(0);
    assertThat(firstY.expression().typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(secondY.expression().typeV2().unwrappedType()).isEqualTo(STR_TYPE);
    assertThat(((UnionType) thirdY.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);

    RegularArgument firstZ = (RegularArgument) calls.get(6).arguments().get(0);
    RegularArgument secondZ = (RegularArgument) calls.get(7).arguments().get(0);
    RegularArgument thirdZ = (RegularArgument) calls.get(8).arguments().get(0);
    assertThat(((UnionType) firstZ.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) secondZ.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) thirdZ.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
  }

  @Test
  void try_except_with_dependents() {
    FileInput fileInput = inferTypes("""
      try:
        x = 42
        y = x
        z = y
        type(x)
        type(y)
        type(z)
      except:
        x = "hello"
        y = x
        z = y
        type(x)
        type(y)
        type(z)
      type(x)
      type(y)
      type(z)
      """);

    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument firstX = (RegularArgument) calls.get(0).arguments().get(0);
    RegularArgument firstY = (RegularArgument) calls.get(1).arguments().get(0);
    RegularArgument firstZ = (RegularArgument) calls.get(2).arguments().get(0);
    assertThat(((UnionType) firstX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) firstY.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) firstZ.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);

    RegularArgument secondX = (RegularArgument) calls.get(3).arguments().get(0);
    RegularArgument secondY = (RegularArgument) calls.get(4).arguments().get(0);
    RegularArgument secondZ = (RegularArgument) calls.get(5).arguments().get(0);
    assertThat(((UnionType) secondX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) secondY.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) secondZ.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);

    RegularArgument thirdX = (RegularArgument) calls.get(6).arguments().get(0);
    RegularArgument thirdY = (RegularArgument) calls.get(7).arguments().get(0);
    RegularArgument thirdZ = (RegularArgument) calls.get(8).arguments().get(0);
    assertThat(((UnionType) thirdX.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) thirdY.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    assertThat(((UnionType) thirdZ.expression().typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
  }

  @Test
  void try_except_list_attributes() {
    FileInput fileInput = inferTypes("""
      try:
        my_list = [1, 2, 3]
        type(my_list)
      except:
        my_list = ["a", "b", "c"]
        type(my_list)
      type(my_list)
      """);

    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument list1 = (RegularArgument) calls.get(0).arguments().get(0);
    RegularArgument list2 = (RegularArgument) calls.get(1).arguments().get(0);
    RegularArgument list3 = (RegularArgument) calls.get(2).arguments().get(0);

    UnionType listType = (UnionType) list1.expression().typeV2();
    assertThat(listType.candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(LIST_TYPE, LIST_TYPE);
    assertThat(listType.candidates())
      .map(ObjectType.class::cast)
      .flatExtracting(ObjectType::attributes)
      .extracting(PythonType::unwrappedType)
      .containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);

    assertThat(list2.expression().typeV2()).isEqualTo(listType);
    assertThat(list3.expression().typeV2()).isEqualTo(listType);

  }

  @Test
  void tryExceptNestedScope() {
    FileInput fileInput = inferTypes("""
      try:
          ...
      except:
          ...
      
      do_smth = None
      
      def something(param):
          type(do_smth)
      """);
    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument doSmthArg = (RegularArgument) calls.get(0).arguments().get(0);
    assertThat(doSmthArg.expression().typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypeForQualifiedExpression() {
    var root = inferTypes("""
      class A:
        def foo():
           ...
      def f():
        a = A()
        a.foo()
      """);

    var fooMethodType = TreeUtils.firstChild(root.statements().statements().get(0), FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::name)
      .map(Expression::typeV2)
      .get();

    var qualifiedExpression = TreeUtils.firstChild(root.statements().statements().get(1), QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .get();

    Assertions.assertThat(qualifiedExpression)
      .isNotNull()
      .extracting(QualifiedExpression::typeV2)
      .isNotNull();

    var qualifiedExpressionType = qualifiedExpression.typeV2();
    Assertions.assertThat(qualifiedExpressionType)
      .isSameAs(fooMethodType)
      .isInstanceOf(FunctionType.class)
      .extracting(PythonType::name)
      .isEqualTo("foo");
  }

  @Test
  void inferTypeForVariableAssignedToQualifiedExpression() {
    var root = inferTypes("""
      class A:
        def foo():
           ...
      def f():
        a = A()
        b = a.foo
        b()
      """);

    var fooMethodType = TreeUtils.firstChild(root.statements().statements().get(0), FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::name)
      .map(Expression::typeV2)
      .get();

    var qualifiedExpression = TreeUtils.firstChild(root.statements().statements().get(1), QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .get();

    Assertions.assertThat(qualifiedExpression)
      .isNotNull()
      .extracting(QualifiedExpression::typeV2)
      .isNotNull();

    var qualifiedExpressionType = qualifiedExpression.typeV2();
    Assertions.assertThat(qualifiedExpressionType)
      .isSameAs(fooMethodType)
      .isInstanceOf(FunctionType.class)
      .extracting(PythonType::name)
      .isEqualTo("foo");

    var bType = TreeUtils.firstChild(root.statements().statements().get(1), FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::body)
      .map(StatementList::statements)
      .map(s -> s.get(2))
      .flatMap(s -> TreeUtils.firstChild(s, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(qualifiedExpressionType).isSameAs(bType);
  }

  @Test
  @Disabled("Member overrides are not supported")
  void inferTypeForOverridenMemberQualifiedExpression() {
    var root = inferTypes("""
      class A:
        def foo():
           ...
      def f():
        a = A()
        a.foo = 42
        a.foo()
      """);


    var fBodyStatements = TreeUtils.firstChild(root.statements().statements().get(1), FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::body)
      .map(StatementList::statements)
      .get();


    var qualifiedExpressionType = TreeUtils.firstChild(fBodyStatements.get(2), QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .map(QualifiedExpression::typeV2)
      .map(PythonType::unwrappedType)
      .get();

    Assertions.assertThat(qualifiedExpressionType)
      .isSameAs(INT_TYPE);
  }


  @Test
  void inferBuiltinsTypeForQualifiedExpression() {
    var root = inferTypes("""
      a = [42]
      a.append(1)
      """);

    var qualifiedExpression = TreeUtils.firstChild(root.statements().statements().get(1), QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .get();

    Assertions.assertThat(qualifiedExpression)
      .isNotNull()
      .extracting(QualifiedExpression::typeV2)
      .isNotNull();

    var builtinsAppendType = LIST_TYPE.resolveMember("append").get();

    var qualifierType = qualifiedExpression.qualifier().typeV2().unwrappedType();
    Assertions.assertThat(qualifierType).isSameAs(LIST_TYPE);

    var qualifiedExpressionType = qualifiedExpression.typeV2();
    Assertions.assertThat(qualifiedExpressionType)
      .isSameAs(builtinsAppendType)
      .isInstanceOf(FunctionType.class)
      .extracting(PythonType::name)
      .isEqualTo("append");
  }

  @Test
  void inferUnknownTypeNestedQualifiedExpression() {
    var root = inferTypes("""
      def f():
        a = foo()
        a.b.c
      """);

    var qualifiedExpression = TreeUtils.firstChild(root.statements().statements().get(0), QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .get();

    Assertions.assertThat(qualifiedExpression)
      .isNotNull()
      .extracting(QualifiedExpression::typeV2)
      .isNotNull();
  }

  @Test
  void inferLoopVarType() {
    var root = inferTypes("""
      def f():
        l = [1,2,3]
        for i in l:
          i
      """);

    var iType = TreeUtils.firstChild(root.statements().statements().get(0), ExpressionStatement.class::isInstance)
      .map(ExpressionStatement.class::cast)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(iType).isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOf(ClassType.class)
      .isEqualTo(INT_TYPE);
  }

  @Test
  void inferLoopOverLiteralVarType() {
    var root = inferTypes("""
      def f():
        for i in [1,2,3]:
          i
      """);

    var iType = TreeUtils.firstChild(root.statements().statements().get(0), ExpressionStatement.class::isInstance)
      .map(ExpressionStatement.class::cast)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(iType).isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOf(ClassType.class)
      .isEqualTo(INT_TYPE);
  }

  @Test
  void inferLoopOverLiteralVarTypeCallable() {
    var root = inferTypes("""
      def foo():
        print("foo")
      
      def bar():
        print("bar")
      
      def f():
        l = [foo, bar]
        for i in l:
          i
      """);

    var fooType = TreeUtils.firstChild(root.statements().statements().get(0), FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::name)
      .map(Expression::typeV2)
      .get();

    var barType = TreeUtils.firstChild(root.statements().statements().get(1), FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::name)
      .map(Expression::typeV2)
      .get();


    var iType = TreeUtils.firstChild(root.statements().statements().get(2), ExpressionStatement.class::isInstance)
      .map(ExpressionStatement.class::cast)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(iType).isInstanceOf(UnionType.class);
    var candidates = ((UnionType) iType).candidates();
    Assertions.assertThat(candidates)
      .allMatch(FunctionType.class::isInstance)
      .contains(fooType, barType);
  }

  @Test
  @Disabled("Need to support collection item type modification")
  void inferLoopOverModifiedList() {
    var root = inferTypes("""
      def f():
        l = [1,2,3]
        l.append("1")
        for i in l:
          i
      """);

    var iType = TreeUtils.firstChild(root, ForStatement.class::isInstance)
      .map(ForStatement.class::cast)
      .map(ForStatement::body)
      .flatMap(b -> TreeUtils.firstChild(b, ExpressionStatement.class::isInstance))
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(iType).isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(INT_TYPE);
  }

  @Test
  @Disabled("Need to support collection item type modification")
  void inferLoopOverFlowSensitivePopulatedItemsList() {
    var root = inferTypes("""
      def f(b):
        a = 1
        if (b):
          a = "1"
        l = [a]
        for i in l:
          i
      """);

    var iType = TreeUtils.firstChild(root, ForStatement.class::isInstance)
      .map(ForStatement.class::cast)
      .map(ForStatement::body)
      .flatMap(b -> TreeUtils.firstChild(b, ExpressionStatement.class::isInstance))
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(iType).isInstanceOf(UnionType.class);

    var candidates = ((UnionType) iType).candidates();
    Assertions.assertThat(candidates)
      .allMatch(ObjectType.class::isInstance)
      .extracting(PythonType::unwrappedType)
      .contains(INT_TYPE, STR_TYPE);
  }

  @Test
  @Disabled("We should consider the declared return type of the __iter__ method here")
  void inferLoopOverCustomIterableVarType() {
    var root = inferTypes("""
      from typing import Iterator
      
      class MyIterable[T]:
        def __iter__(self) -> Iterator[str]:
          ...
      
      def f():
        a = MyIterable[int]()
        for i in a:
          i
      """);

    var iType = TreeUtils.firstChild(root, ForStatement.class::isInstance)
      .map(ForStatement.class::cast)
      .map(ForStatement::body)
      .flatMap(b -> TreeUtils.firstChild(b, ExpressionStatement.class::isInstance))
      .map(ExpressionStatement.class::cast)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(iType).isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOf(ClassType.class)
      .isEqualTo(STR_TYPE);
  }


  @Test
  void inferLoopOverUnknownVarType() {
    var root = inferTypes("""
      def f(l):
        for i in l:
          i
      """);

    var iType = TreeUtils.firstChild(root.statements().statements().get(0), ExpressionStatement.class::isInstance)
      .map(ExpressionStatement.class::cast)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(iType).isInstanceOf(UnknownType.class)
      .isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void inferLoopOverNonIterableVarType() {
    var root = inferTypes("""
      def f():
        l = 1
        for i in l:
          i
      """);

    var iType = TreeUtils.firstChild(root.statements().statements().get(0), ExpressionStatement.class::isInstance)
      .map(ExpressionStatement.class::cast)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(iType).isInstanceOf(UnknownType.class)
      .isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void inferLoopOverNonIterableVarTypeInTry() {
    var root = inferTypes("""
      def f():
        try:
          l = 1
          for i in l:
            i
        except:
          ...
      """);

    var iType = TreeUtils.firstChild(root.statements().statements().get(0), ExpressionStatement.class::isInstance)
      .map(ExpressionStatement.class::cast)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(iType).isInstanceOf(UnknownType.class)
      .isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void hasUnknownListItemType() {
    var root = inferTypes("""
      def f(i):
        l = [1,2,i]
      """);

    var lType = TreeUtils.firstChild(root.statements().statements().get(0), AssignmentStatement.class::isInstance)
      .map(AssignmentStatement.class::cast)
      .map(AssignmentStatement::lhsExpressions)
      .map(Collection::stream)
      .flatMap(Stream::findFirst)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(lType).isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("list");

    Assertions.assertThat(lType)
      .extracting(ObjectType.class::cast)
      .extracting(ObjectType::attributes)
      .asList()
      .hasSize(1)
      .contains(PythonType.UNKNOWN);

  }

  @Test
  @Disabled("Attribute types resolving")
  void inferBuiltinsAttributeTypeForQualifiedExpression() {
    var root = inferTypes("""
      def f():
        e = OSError()
        e.errno
      """);

    var qualifiedExpression = TreeUtils.firstChild(root.statements().statements().get(0), QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .get();

    Assertions.assertThat(qualifiedExpression)
      .isNotNull()
      .extracting(QualifiedExpression::typeV2)
      .isNotNull();

    var qualifiedExpressionType = qualifiedExpression.typeV2();
    Assertions.assertThat(qualifiedExpressionType)
      .isInstanceOf(ObjectType.class)
      .extracting(ObjectType.class::cast)
      .extracting(ObjectType::type)
      .extracting(PythonType::name)
      .isEqualTo("int");
  }

  @Test
  void conditionallyAssignedString() {
    var root = inferTypes("""
      if cond:
        x = "hello"
      else:
        x = "world"
      x
      """);

    var xType = TreeUtils.firstChild(root.statements().statements().get(1), Name.class::isInstance)
      .map(Name.class::cast)
      .map(Name::typeV2)
      .get();

    Assertions.assertThat(xType).extracting(PythonType::unwrappedType).isSameAs(STR_TYPE);
  }

  @Test
  void inferClassHierarchyHasMetaClass() {
    var root = inferTypes("""
      class CustomMetaClass:
        ...
      
      class ParentClass(metaclass=CustomMetaClass):
        ...
      
      class ChildClass(ParentClass):
        ...
      
      def f():
        a = ChildClass()
        a
      """);


    var childClassType = TreeUtils.firstChild(root.statements().statements().get(2), ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .map(ClassDef::name)
      .map(Expression::typeV2)
      .map(ClassType.class::cast)
      .get();

    Assertions.assertThat(childClassType.hasMetaClass()).isTrue();

    var aType = TreeUtils.firstChild(root.statements().statements().get(3), ExpressionStatement.class::isInstance)
      .map(ExpressionStatement.class::cast)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(aType)
      .isNotNull()
      .isNotEqualTo(PythonType.UNKNOWN)
      .extracting(PythonType::unwrappedType)
      .isSameAs(childClassType);
  }

  @Test
  void inferReassignedParameterType() {
    var root = inferTypes("""
      def reassigned_param(a, param):
          param = 1
          if a:
              param = "1"
          param()
      """);

    var paramType = TreeUtils.firstChild(root, CallExpression.class::isInstance)
      .map(CallExpression.class::cast)
      .map(CallExpression::callee)
      .map(Expression::typeV2)
      .map(UnionType.class::cast)
      .get();


    Assertions.assertThat(paramType).isInstanceOf(UnionType.class);
    Assertions.assertThat(paramType.candidates())
      .hasSize(2);

    var candidatesUnwrappedType = paramType.candidates().stream()
      .map(PythonType::unwrappedType)
      .toList();

    Assertions.assertThat(candidatesUnwrappedType)
      .contains(INT_TYPE, STR_TYPE);
  }

  @Test
  void inferConditionallyReassignedParameterType() {
    var root = inferTypes("""
      def reassigned_param(a, param):
          if a:
              param = "1"
          param()
      """);

    var paramType = TreeUtils.firstChild(root, CallExpression.class::isInstance)
      .map(CallExpression.class::cast)
      .map(CallExpression::callee)
      .map(Expression::typeV2)
      .get();

    Assertions.assertThat(paramType).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void list_comprehension() {
    assertThat(lastExpression(
      """
      x = [a for a in foo()]
      x
      """
    ).typeV2().unwrappedType()).isEqualTo(LIST_TYPE);
  }

  @Test
  void set_comprehension() {
    assertThat(lastExpression(
      """
      x = {a for a in foo()}
      x
      """
    ).typeV2().unwrappedType()).isEqualTo(SET_TYPE);
  }

  @Test
  void dict_comprehension() {
    assertThat(lastExpression(
      """
      x = {num: num**2 for num in numbers()}
      x
      """
    ).typeV2().unwrappedType()).isEqualTo(DICT_TYPE);
  }

  @Test
  void comprehension_if() {
    assertThat(lastExpression(
      """
      x = [num for num in numbers if num % 2 == 0]
      x
      """
    ).typeV2().unwrappedType()).isEqualTo(LIST_TYPE);
  }

  @Test
  void generator_expression() {
    assertThat(lastExpression(
      """
      x = (num**2 for num in numbers())
      x
      """
    ).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void return_type_of_call_expression_1() {
    assertThat(lastExpression(
      """
      x = [1,2,3]
      a = x.append(42)
      a
      """
    ).typeV2().unwrappedType()).isEqualTo(NONE_TYPE);
  }

  @Test
  void return_type_of_call_expression_2() {
    assertThat(lastExpression(
      """
      x = [1,2,3]
      a = x.sort()
      a
      """
    ).typeV2().unwrappedType()).isEqualTo(NONE_TYPE);
  }

  @Test
  void return_type_of_call_expression_union_type() {
    FileInput fileInput = inferTypes(
      """
        class A:
          def foo(self): ...
        class B:
          def bar(self): ...
        a = A
        b = B
        if cond:
          x = a
        else:
          x = b
        y = x()
        y
        """
    );
    var classA = TreeUtils.firstChild(fileInput.statements().statements().get(0), ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .map(ClassDef::name)
      .map(Expression::typeV2)
      .map(ClassType.class::cast)
      .get();

    var classB = TreeUtils.firstChild(fileInput.statements().statements().get(1), ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .map(ClassDef::name)
      .map(Expression::typeV2)
      .map(ClassType.class::cast)
      .get();

    assertThat(((ExpressionStatement) fileInput.statements().statements().get(6)).expressions().get(0).typeV2()).isInstanceOf(ObjectType.class);
    UnionType unionType = (UnionType) ((ExpressionStatement) fileInput.statements().statements().get(6)).expressions().get(0).typeV2().unwrappedType();
    assertThat(unionType.candidates()).containsExactlyInAnyOrder(classA, classB);
  }

  @Test
  void imported_symbol_call_return_type() {
    assertThat(lastExpression(
      """
      import fcntl
      ret = fcntl.flock(..., ...)
      ret
      """
    ).typeV2().unwrappedType()).isEqualTo(NONE_TYPE);
  }

  @Test
  void basic_imported_symbol() {
    assertThat(lastExpression(
      """
      import fcntl
      fcntl
      """
    ).typeV2()).isInstanceOf(ModuleType.class);
  }

  @Test
  void resolvedBuiltinLazyType() {
    FileInput fileInput = inferTypes("""
      copyright
      """);
    FunctionType functionType = ((FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0).typeV2());
    PythonType pythonType = functionType.returnType();
    assertThat(pythonType.unwrappedType()).isEqualTo(NONE_TYPE);
  }

  @Test
  void resolvedTypingLazyType() {
    FileInput fileInput = inferTypes("""
      import calendar
      calendar.Calendar.iterweekdays
      """);
    FunctionType functionType = ((FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    PythonType returnType = functionType.returnType();
    assertThat(returnType).isInstanceOf(ClassType.class);
    assertThatThrownBy(() -> functionType.resolveLazyReturnType(PythonType.UNKNOWN))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Trying to resolve an already resolved lazy type.");
  }

  @Test
  void resolveCustomTypeLazyType() {
    FileInput fileInput = inferTypes("""
      import ldap
      connect = ldap.initialize('ldap://127.0.0.1:1389')
      connect
      """);
    ObjectType pythonType = (ObjectType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(pythonType.unwrappedType()).extracting(PythonType::name).isEqualTo("LDAPObject");
  }

  @Test
  @Disabled("SONARPY-2039")
  void lazyTypeResolutionForModules() {
    // SONARPY-2039: The unknown "http" member of "django" is a PythonType.UNKNOWN that is manually replaced after being resolved
    // It should instead be a LazyType that is scheduled for a clean resolution
    FileInput fileInput = inferTypes("""
      from django.http import request
      request
      """);
    PythonType pythonType = (PythonType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(pythonType).isInstanceOf(ModuleType.class);
  }

  @Test
  void resolveIncorrectLazyType() {
    ProjectLevelSymbolTable empty = ProjectLevelSymbolTable.empty();
    TypeShed typeShed = new TypeShed(empty);

    SymbolsModuleTypeProvider symbolsModuleTypeProvider = new SymbolsModuleTypeProvider(empty, typeShed);
    ModuleType builtinModule = symbolsModuleTypeProvider.createBuiltinModule();
    symbolsModuleTypeProvider.createModuleType(List.of("typing"), builtinModule);
    assertThat(symbolsModuleTypeProvider.resolveLazyType(new LazyType("unknown", symbolsModuleTypeProvider))).isEqualTo(PythonType.UNKNOWN);
    assertThat(symbolsModuleTypeProvider.resolveLazyType(new LazyType("typing.unknown", symbolsModuleTypeProvider))).isEqualTo(PythonType.UNKNOWN);
    assertThat(symbolsModuleTypeProvider.resolveLazyType(new LazyType("unrelated.unknown", symbolsModuleTypeProvider))).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void resolveIncorrectLazyType2() {
    ProjectLevelSymbolTable empty = ProjectLevelSymbolTable.empty();
    TypeShed typeShed = new TypeShed(empty);

    SymbolsModuleTypeProvider symbolsModuleTypeProvider = new SymbolsModuleTypeProvider(empty, typeShed);
    ModuleType builtinModule = symbolsModuleTypeProvider.createBuiltinModule();
    symbolsModuleTypeProvider.createModuleType(List.of("typing"), builtinModule);

    ClassSymbol symbol = Mockito.mock(ClassSymbolImpl.class);
    Mockito.when(symbol.kind()).thenReturn(Symbol.Kind.OTHER);
    assertThat(symbolsModuleTypeProvider.resolvePossibleLazyType(new HashMap<>(), symbol, "typing.Iterable.unknown")).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void imported_symbol_in_different_branch() {
    FileInput fileInput = inferTypes("""
      if x:
        import fcntl
      def lock():
        fcntl
      """);
    Statement functionDef = fileInput.statements().statements().get(1);
    ExpressionStatement fcntlStatement = ((ExpressionStatement) TreeUtils.firstChild(functionDef, t -> t.is(Tree.Kind.EXPRESSION_STMT)).get());
    assertThat(fcntlStatement.expressions().get(0).typeV2()).isInstanceOf(ModuleType.class);
    assertThat(fcntlStatement.expressions().get(0).typeV2().name()).isEqualTo("fcntl");
  }

  @Test
  void basic_imported_symbols() {
    FileInput fileInput = inferTypes(
      """
      import fcntl, math
      fcntl
      math
      """
    );
    PythonType fnctlModule = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(fnctlModule).isInstanceOf(ModuleType.class);
    assertThat(fnctlModule.name()).isEqualTo("fcntl");
    PythonType mathModule = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(mathModule).isInstanceOf(ModuleType.class);
    assertThat(mathModule.name()).isEqualTo("math");
    assertThat(((UnionType) mathModule.resolveMember("acos").get()).candidates()).allMatch(FunctionType.class::isInstance);
  }

  @Test
  void isInstanceTests() {
    var xType = lastExpression("""
      def foo(x: int):
        if isinstance(x, Foo):
          ...
        x
      """).typeV2();
    Assertions.assertThat(xType).isSameAs(PythonType.UNKNOWN);

    xType = lastExpression("""
      def foo(x: int):
        if isinstance(x):
          ...
        x
      """).typeV2();
    Assertions.assertThat(xType.unwrappedType()).isSameAs(INT_TYPE);

    xType = lastExpression("""
      def foo(x: int):
        if isinstance(x.b, Foo):
          ...
        x
      """).typeV2();
    Assertions.assertThat(xType.unwrappedType()).isSameAs(INT_TYPE);

    xType = lastExpression("""
      def foo():
        x = 10
        if isinstance(x, Foo):
          ...
        x
      """).typeV2();
    Assertions.assertThat(xType.unwrappedType()).isSameAs(INT_TYPE);

    xType = lastExpression("""
      def foo(x: list):
        if isinstance(**x, Foo):
          ...
        x
      """).typeV2();

    Assertions.assertThat(xType.unwrappedType()).isSameAs(LIST_TYPE);
  }

  private static FileInput inferTypes(String lines) {
    return inferTypes(lines, PROJECT_LEVEL_TYPE_TABLE);
  }

  private static FileInput inferTypes(String lines, ProjectLevelTypeTable projectLevelTypeTable) {
    FileInput root = parse(lines);

    var symbolTable = new SymbolTableBuilderV2(root)
      .build();
    new TypeInferenceV2(projectLevelTypeTable, pythonFile, symbolTable).inferTypes(root);
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
