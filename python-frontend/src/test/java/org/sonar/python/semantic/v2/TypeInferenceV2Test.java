/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.semantic.v2;

import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Stream;
import org.assertj.core.api.InstanceOfAssertFactories;
import org.assertj.core.data.Index;
import org.assertj.core.groups.Tuple;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.AwaitExpression;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.Member;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.plugins.python.api.types.v2.UnknownType.UnresolvedImportType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.tree.ExpressionStatementImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.tree.TupleImpl;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.LazyTypeWrapper;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.getFirstDescendant;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.types.v2.TypesTestUtils.BOOL_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.DICT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.EXCEPTION_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.FLOAT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.FROZENSET_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.LIST_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.NONE_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.SET_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.STR_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.TUPLE_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.TYPE_TYPE;

public class TypeInferenceV2Test {

  static PythonFile pythonFile = pythonFile("mod");

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
  void builtinGenericType() {
    Expression expression = lastExpression(
      """
        x = list[str]()
        x
        """
    );
    assertThat(expression.typeV2().unwrappedType()).isEqualTo(LIST_TYPE);
  }

  @Test
  void userDefinedGenericType() {
    FileInput fileInput = inferTypes(
      """
        from typing import Generic, TypeVar
        T = TypeVar('T')
        class MyClass(Generic[T]): ...
        x = MyClass[str]()
        x
        """
    );
    PythonType classType = ((ClassDef) fileInput.statements().statements().get(2)).name().typeV2();
    ObjectType xType = (ObjectType) ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isEqualTo(classType);
    // SONARPY-2356: Instantiation of specialized classes
    assertThat(xType.attributes()).isEmpty();
  }

  @Test
  void inheritedGenericType() {
    FileInput fileInput = inferTypes(
      """
        from typing import Generic, TypeVar
        T = TypeVar('T')
        class MyClass(Generic[T]): ...
        class MyOtherClass(MyClass[T]): ...
        x = MyOtherClass[str]()
        x
        """
    );
    ClassType myOtherClassType = (ClassType) ((ClassDef) fileInput.statements().statements().get(3)).name().typeV2();
    assertThat(myOtherClassType.isGeneric()).isTrue();
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(5)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isEqualTo(myOtherClassType);
  }

  @Test
  void inheritedGenericTypeUnsupportedExpression() {
    FileInput fileInput = inferTypes(
      """
        from typing import Generic, TypeVar
        T = TypeVar('T')
        class MyClass(Generic[T()]): ...
        class MyOtherClass(MyClass[T]): ...
        x = MyOtherClass[str]()
        x
        """
    );
    ClassType myOtherClassType = (ClassType) ((ClassDef) fileInput.statements().statements().get(3)).name().typeV2();
    assertThat(myOtherClassType.isGeneric()).isFalse();
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(5)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isInstanceOf(UnknownType.class);
  }

  @Test
  void inheritedGenericTypeVarAnnotatedAssignment() {
    FileInput fileInput = inferTypes(
      """
        from typing import Generic, TypeVar
        T: TypeVar = TypeVar('T')
        class MyClass(Generic[T]): ...
        class MyOtherClass(MyClass[T]): ...
        x = MyOtherClass[str]()
        x
        """
    );
    ClassType myOtherClassType = (ClassType) ((ClassDef) fileInput.statements().statements().get(3)).name().typeV2();
    assertThat(myOtherClassType.isGeneric()).isTrue();
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(5)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isEqualTo(myOtherClassType);
  }

  @Test
  void inheritedGenericTypeVarAssignmentExpression() {
    FileInput fileInput = inferTypes(
      """
        from typing import Generic, TypeVar
        foo(T:=TypeVar('T'))
        class MyClass(Generic[T]): ...
        class MyOtherClass(MyClass[T]): ...
        x = MyOtherClass[str]()
        x
        """
    );
    ClassType myOtherClassType = (ClassType) ((ClassDef) fileInput.statements().statements().get(3)).name().typeV2();
    assertThat(myOtherClassType.isGeneric()).isTrue();
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(5)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isEqualTo(myOtherClassType);
  }

  @Test
  void inheritedGenericTypeUndefinedTypeVar() {
    FileInput fileInput = inferTypes(
      """
        from typing import Generic
        class MyClass(Generic[T]): ...
        class MyOtherClass(MyClass[T]): ...
        x = MyOtherClass[str]()
        x
        """
    );
    ClassType myOtherClassType = (ClassType) ((ClassDef) fileInput.statements().statements().get(2)).name().typeV2();
    // TypeVar is undefined: not a proper generic
    assertThat(myOtherClassType.isGeneric()).isFalse();
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isInstanceOf(UnknownType.class);
  }

  @Test
  void inheritedGenericTypeNoSubscription() {
    FileInput fileInput = inferTypes(
      """
        from typing import Generic
        class MyClass(Generic[T]): ...
        class MyOtherClass(MyClass): ...
        # Note: this actually throws a TypeError
        x = MyOtherClass[str]()
        x
        """
    );
    ClassType myOtherClassType = (ClassType) ((ClassDef) fileInput.statements().statements().get(2)).name().typeV2();
    // MyOtherClass can no longer be considered generic (non-generic subclass)
    assertThat(myOtherClassType.isGeneric()).isFalse();
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isInstanceOf(UnknownType.UnknownTypeImpl.class);
  }

  @Test
  void inheritedGenericTypeConcreteType() {
    FileInput fileInput = inferTypes(
      """
        from typing import Generic
        class MyClass(Generic[T]): ...
        class MyOtherClass(MyClass[str]): ...
        # Note: this instantiation actually throws a TypeError
        x = MyOtherClass[str]()
        x
        """
    );
    ClassType myOtherClassType = (ClassType) ((ClassDef) fileInput.statements().statements().get(2)).name().typeV2();
    // SONARPY-2356: MyOtherClass can no longer be considered generic (specialized version of MyClass)
    assertThat(myOtherClassType.isGeneric()).isFalse();
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isInstanceOf(UnknownType.class);
  }

  @Test
  void basicGenericTypeParameter() {
    FileInput fileInput = inferTypes(
      """
        class MyClass[T](): ...
        x = MyClass[str]()
        x
        """
    );
    ClassType myOtherClassType = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    assertThat(myOtherClassType.isGeneric()).isTrue();
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isEqualTo(myOtherClassType);
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
      .extracting(UnresolvedImportType.class::cast)
      .extracting(UnresolvedImportType::importPath)
      .isEqualTo("something");
    assertThat(importedNames.get(1))
      .extracting(Expression::typeV2)
      .extracting(UnresolvedImportType.class::cast)
      .extracting(UnresolvedImportType::importPath)
      .isEqualTo("something.unknown");
  }

  @Test
  void testInheritanceFromUnresolvedImports() {
    FileInput root = inferTypes("""
      from unknown import Parent
      class A(Parent): ...
      """);

    var classDef = (ClassDef) root.statements().statements().get(1);
    assertThat(classDef.args().arguments().get(0))
      .extracting(RegularArgument.class::cast)
      .extracting(RegularArgument::expression)
      .extracting(Expression::typeV2)
      .extracting(UnresolvedImportType.class::cast)
      .extracting(UnresolvedImportType::importPath)
      .isEqualTo("unknown.Parent");
  }

  @Test
  void testUnresolvedImportTypePropagationInsideFunctions() {
    var fileInput = inferTypes("""
      from a import b
      def function():
        f(b)
      """);
    var functionDef = (FunctionDef) fileInput.statements().statements().get(1);
    var funcCall = ((ExpressionStatement) functionDef.body().statements().get(0)).expressions().get(0);
    var arg = ((RegularArgument) ((CallExpression) funcCall).arguments().get(0));
    var argType = arg.expression().typeV2();

    assertThat(argType).isInstanceOfSatisfying(UnresolvedImportType.class, a -> assertThat(a.importPath()).isEqualTo("a.b"));
  }

  @Test
  void testRelativeImport() {
    FileInput fileInput = inferTypes("""
      from . import module
      module
      """);

    PythonType pythonType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(pythonType).isInstanceOf(UnresolvedImportType.class);
    assertThat(((UnresolvedImportType) pythonType).importPath()).isEqualTo("my_package.module");

    fileInput = inferTypes("""
      from .. import module
      module
      """);
    pythonType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(pythonType).isInstanceOf(UnresolvedImportType.class);
    assertThat(((UnresolvedImportType) pythonType).importPath()).isEqualTo("module");

    fileInput = inferTypes("""
      from .hello import module
      module
      """);
    pythonType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(pythonType).isInstanceOf(UnresolvedImportType.class);
    assertThat(((UnresolvedImportType) pythonType).importPath()).isEqualTo("my_package.hello.module");

    fileInput = inferTypes("""
      from ..second_hello import module
      module
      """);
    pythonType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(pythonType).isInstanceOf(UnresolvedImportType.class);
    assertThat(((UnresolvedImportType) pythonType).importPath()).isEqualTo("second_hello.module");
  }

  @Test
  void testProjectLevelSymbolTableImports() {
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.from(
      Map.of("something.known", Collections.singleton(new ClassDescriptor.ClassDescriptorBuilder().withName("C").withFullyQualifiedName("something.known.C").build()), "something",
        Set.of())));

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
      .hasFieldOrPropertyWithValue("fullyQualifiedName", "something.known")
      .hasFieldOrPropertyWithValue("name", "known")
      .matches(type -> {
        assertThat(type.resolveMember("C").get())
          .isInstanceOf(ClassType.class);
        return true;
      })
    ;
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

    assertThat(type)
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

    assertThat(type1)
      .isEqualTo(PythonType.UNKNOWN);

    assertThat(type2)
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
  void child_class_method_call_is_not_a_member_of_parent_class_type() {
    FileInput fileInput = inferTypes("""
      class A:
        def meth(self):
          return self.foo()
      class B(A):
        def foo(self):
          ...
      """
    );
    // SONARPY-2327 The method call to foo() in class A is a member of ClassSymbol A but not a member of ClassType A.
    Optional<ClassSymbol> classSymbolA = fileInput.globalVariables().stream().filter(s -> s.name().equals("A")).map(ClassSymbol.class::cast).findFirst();
    assertThat(classSymbolA).isPresent();
    assertThat(classSymbolA.get().canHaveMember("foo")).isTrue();
    assertThat(classSymbolA.get().declaredMembers()).extracting("kind", "name")
      .containsExactlyInAnyOrder(Tuple.tuple(Symbol.Kind.FUNCTION, "meth"), Tuple.tuple(Symbol.Kind.OTHER, "foo"));

    ClassType classTypeA = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    assertThat(classTypeA.members().stream().anyMatch(m -> m.name().equals("foo"))).isFalse();
    assertThat(classTypeA.members().stream().anyMatch(m -> m.name().equals("meth"))).isTrue();
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
  void nestedClassDefinitions() {
    Expression expr = lastExpression("""
      class A:
        class B:
          pass
      A.B
      """);
    assertThat(expr.typeV2())
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("B");

    Expression expr2 = lastExpression("""
      class A:
        class B:
          class C:
            pass
      A.B.C
      """);
    assertThat(expr2.typeV2())
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("C");

    Expression expr3 = lastExpression("""
      class A:
        class B:
          pass
        B = 42
      A.B
      """);
    assertThat(expr3.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void nestedClassDefinitionsWithInheritance() {
    Expression exprWithInheritance = lastExpression("""
      class A:
        class B:
          pass
      class C(A):
        pass
      C.B
      """);
    assertThat(exprWithInheritance.typeV2())
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("B");

    Expression exprWithMultiInheritance = lastExpression("""
      class A:
        class B:
          pass
      class C: pass
      class D(C, A):
        pass
      D.B
      """);
    assertThat(exprWithMultiInheritance.typeV2())
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("B");

    Expression exprWithMultiInheritance2 = lastExpression("""
      class A:
        pass
      class C:
        class B:
          pass
      class D(C, A):
        pass
      D.B
      """);
    assertThat(exprWithMultiInheritance2.typeV2())
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("B");
  }

  @Test
  void multiInheritanceConflicts() {
    Expression exprWithMultiInheritance1 = lastExpression("""
      class A: B = "hi"
      class C:
        class B: pass
      class D(A, C): pass
      D.B
      """);
    assertThat(exprWithMultiInheritance1.typeV2())
      .isInstanceOf(UnknownType.class);

    Expression exprWithMultiInheritance2 = lastExpression("""
      class A:
        class B: pass
      class C:
        B = "hi"
      class D(A, C): pass
      D.B
      """);
    assertThat(exprWithMultiInheritance2.typeV2())
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("B");
  }

  @Test
  void annotatedClassFieldsInClassDefinition() {
    var expression = lastExpression(
      """
        class A:
          a : str = "hi"
          b : int
        A
        """
    );
    assertThat(expression.typeV2()).isInstanceOfSatisfying(ClassType.class, classType -> {
      assertThat(classType.members()).hasSize(2);
      var aFieldMember = classType.members().stream().filter(m -> "a".equals(m.name())).findFirst().get();
      var bFieldMember = classType.members().stream().filter(m -> "b".equals(m.name())).findFirst().get();
      assertThat(aFieldMember.name()).isEqualTo("a");
      assertThat(aFieldMember.type()).isInstanceOf(ObjectType.class).extracting(PythonType::unwrappedType).isEqualTo(STR_TYPE);
      assertThat(bFieldMember.name()).isEqualTo("b");
      assertThat(bFieldMember.type()).isInstanceOf(ObjectType.class).extracting(PythonType::unwrappedType).isEqualTo(INT_TYPE);
    });
  }

  @Test
  void annotatedClassFieldsInClassDefinitionSymbol() {
    var tree = parseWithoutSymbols(
      """
        class A:
          a : str = "hi"
          b : int
        """
    );
    var projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));

    var symbol = (ClassSymbol) projectLevelSymbolTable.getSymbol("mod.A");
    var aFieldSymbol = symbol.resolveMember("a").get();
    var bFieldSymbol = symbol.resolveMember("b").get();
    assertThat(aFieldSymbol.name()).isEqualTo("a");
    assertThat(aFieldSymbol.annotatedTypeName()).isEqualTo("str");
    assertThat(bFieldSymbol.name()).isEqualTo("b");
    assertThat(bFieldSymbol.annotatedTypeName()).isEqualTo("int");
  }

  @Test
  void annotatedClassFieldOverrideInClassDefinition() {
    var expression = lastExpression(
      """
        class A:
          a : str
          a : int
        A
        """
    );
    assertThat(expression.typeV2()).isInstanceOfSatisfying(ClassType.class, classType -> {
      assertThat(classType.members()).hasSize(1);
      var aFieldMember = classType.members().stream().filter(m -> "a".equals(m.name())).findFirst().get();
      assertThat(aFieldMember.name()).isEqualTo("a");
      assertThat(aFieldMember.type()).isInstanceOf(ObjectType.class).extracting(PythonType::unwrappedType).isEqualTo(INT_TYPE);
    });
  }

  @Test
  void staticFieldsInClassDefinition() {
    Expression expr = lastExpression("""
      class A:
        test = "hi"
      A
      """);
    assertThat(expr.typeV2()).isInstanceOfSatisfying(ClassType.class, classType -> {
      assertThat(classType.members()).hasSize(1);
      var member = classType.members().iterator().next();
      assertThat(member.name()).isEqualTo("test");
      assertThat(member.type()).isEqualTo(PythonType.UNKNOWN);
    });

    Expression expr2 = lastExpression("""
      class A:
        test = "hi"
        test = True
      A
      """);
    assertThat(expr2.typeV2()).isInstanceOfSatisfying(ClassType.class, classType -> {
      assertThat(classType.members()).hasSize(1);
      var member = classType.members().iterator().next();
      assertThat(member.name()).isEqualTo("test");
      assertThat(member.type()).isEqualTo(PythonType.UNKNOWN);
    });

    Expression expr3 = lastExpression("""
      class A:
        test = classmethod(...)
      A
      """);
    assertThat(expr3.typeV2()).isInstanceOfSatisfying(ClassType.class, classType -> {
      assertThat(classType.members()).hasSize(1);
      var member = classType.members().iterator().next();
      assertThat(member.name()).isEqualTo("test");
      assertThat(member.type()).isEqualTo(PythonType.UNKNOWN);
    });
  }

  @Test
  void staticFieldsInInheritedClasses() {
    Expression exprWithInheritance = lastExpression("""
      class A:
        test = "hi"
      class B(A):
        pass
      B
      """);
    assertThat(exprWithInheritance.typeV2()).isInstanceOfSatisfying(ClassType.class, classType -> {
      var member = classType.resolveMember("test").get();
      assertThat(member).isEqualTo(PythonType.UNKNOWN);
    });

    Expression exprWithInheritance2 = lastExpression("""
      class A:
        test = True
      class B(A):
        test = "hi"
      class C(B):
        pass
      C
      """);
    assertThat(exprWithInheritance2.typeV2()).isInstanceOfSatisfying(ClassType.class, classType -> {
      var member = classType.resolveMember("test").get();
      assertThat(member).isEqualTo(PythonType.UNKNOWN);
    });

    Expression exprWithMultiInheritance = lastExpression("""
      class A:
          test = "hi"
      class B: pass
      class C(A, B):
        pass
      C
      """);
    assertThat(exprWithMultiInheritance.typeV2()).isInstanceOfSatisfying(ClassType.class, classType -> {
      var member = classType.resolveMember("test").get();
      assertThat(member).isEqualTo(PythonType.UNKNOWN);
    });

    Expression exprWithMultiInheritance2 = lastExpression("""
      class A:
        pass
      class B:
        test = "hi"
      class C(A, B):
        pass
      C
      """);
    assertThat(exprWithMultiInheritance2.typeV2()).isInstanceOfSatisfying(ClassType.class, classType -> {
      var member = classType.resolveMember("test").get();
      assertThat(member).isEqualTo(PythonType.UNKNOWN);
    });

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
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
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
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
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
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void inferTypesInsideFunction4() {
    FileInput root = inferTypes("""
      def foo():
        x = 42
      x
      """);

    var lastExpressionStatement = (ExpressionStatement) root.statements().statements().get(root.statements().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction5() {
    FileInput root = inferTypes("""
      def foo(param: int):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().typeSource()).isEqualTo(TypeSource.TYPE_HINT);

    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void typeSourceOfCallExpressionResultDependsOnTypeSourceOfQualifier() {
    FileInput root = inferTypes("""
      def foo(x: int):
        y = x.conjugate()
        y
        z = x.conjugate().conjugate()
        z
      """);
    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var yStatement = (ExpressionStatement) functionDef.body().statements().get(1);
    PythonType yType = yStatement.expressions().get(0).typeV2();
    assertThat(yType).isInstanceOf(ObjectType.class);
    assertThat(yType.unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(yType.typeSource()).isEqualTo(TypeSource.TYPE_HINT);

    var zStatement = (ExpressionStatement) functionDef.body().statements().get(3);
    PythonType zType = zStatement.expressions().get(0).typeV2();
    assertThat(zType).isInstanceOf(ObjectType.class);
    assertThat(zType.unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(zType.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
  }

  @Test
  void typeSourceOfCallExpressionResultDependsOnTypeSourceOfName() {
    FileInput fileInput = inferTypes("""
      from pyasn1.debug import Printer
      def foo(p: Printer):
        a = p()
        a
        b = p.__call__()
        b
      """);

    var functionDef = (FunctionDef) fileInput.statements().statements().get(1);
    var aStatement = (ExpressionStatement) functionDef.body().statements().get(1);
    PythonType aType = aStatement.expressions().get(0).typeV2();
    assertThat(aType).isInstanceOf(ObjectType.class);
    assertThat(aType.unwrappedType()).isEqualTo(NONE_TYPE);
    assertThat(aType.typeSource()).isEqualTo(TypeSource.TYPE_HINT);

    var bStatement = (ExpressionStatement) functionDef.body().statements().get(3);
    PythonType bType = bStatement.expressions().get(0).typeV2();
    assertThat(bType).isInstanceOf(ObjectType.class);
    assertThat(bType.unwrappedType()).isEqualTo(NONE_TYPE);
    assertThat(bType.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
  }

  @Test
  void typeSourceIsExactByDefault() {
    // Test that when calling a function from stub, the return type is computed correctly
    FileInput root = inferTypes("""
      def foo() -> int: ...
      foo()
      """);
    CallExpression callExpression = ((CallExpression) ((ExpressionStatement) root.statements().statements().get(1)).expressions().get(0));
    var resultType = callExpression.typeV2();
    assertThat(resultType).isInstanceOf(ObjectType.class);
    assertThat(resultType.unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void inferTypesInsideFunction6() {
    FileInput root = inferTypes("""
      def foo(param: int):
        param = "hello"
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(STR_TYPE);
  }

  @Test
  void inferTypesInsideFunction7() {
    FileInput root = inferTypes("""
      def foo(param):
        param = "hello"
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(STR_TYPE);
  }

  @Test
  void inferTypesInsideFunction8() {
    FileInput root = inferTypes("""
      def foo(param: int):
        x = param
        x
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().typeSource()).isEqualTo(TypeSource.TYPE_HINT);
  }

  @Test
  void inferTypesInsideFunction9() {
    FileInput root = inferTypes("""
      def foo(param: list[int]):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    var type = (ObjectType) lastExpressionStatement.expressions().get(0).typeV2();
    assertThat(type.unwrappedType()).isEqualTo(LIST_TYPE);
    assertThat(type.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
    assertThat(type.attributes())
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
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction11() {
    FileInput root = inferTypes("""
      def foo(param: something_unknown[int]):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction12() {
    FileInput root = inferTypes("""
      o = "123"
      def foo(param: o):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(1);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferTypesInsideFunction13() {
    FileInput root = inferTypes("""
      def foo(param: int | str):
        param
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    var type = (UnionType) lastExpressionStatement.expressions().get(0).typeV2();

    assertThat(type.candidates())
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

    assertThat(firstType.unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(lastType.unwrappedType()).isEqualTo(STR_TYPE);
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

    assertThat(lastType.unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void inferFunctionParameterTypes() {
    FileInput root = inferTypes("""
      def foo(param: int, *args, **kwargs):
        ...
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(0);
    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(1).declaredType().type().unwrappedType()).isEqualTo(TUPLE_TYPE);
    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(2).declaredType().type().unwrappedType()).isEqualTo(DICT_TYPE);
  }

  @Test
  void inferFunctionParameterTypes2() {
    FileInput root = inferTypes("""
      class A: ...
      class A: ...
      def foo(param: A) -> A:
        ...
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(2);
    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
    assertThat(((FunctionType) functionDef.name().typeV2()).returnType().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferFunctionParameterTypes3() {
    FileInput root = inferTypes("""
      class A:
        def foo():
          ...
      def foo(param: A) -> A:
        ...
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(1);
    var classType = ((ClassDef) root.statements().statements().get(0)).name().typeV2();

    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(classType);
    assertThat(((FunctionType) functionDef.name().typeV2()).returnType().unwrappedType()).isEqualTo(classType);
  }

  @Test
  void inferFunctionParameterTypes4() {
    FileInput root = inferTypes("""
      from re import Pattern
      Pattern
      def foo(param: Pattern) -> Pattern:
        ...
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(2);
    var patternType = ((Name) ((ExpressionStatementImpl) root.statements().statements().get(1)).expressions().get(0)).typeV2();

    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(patternType);
    assertThat(((FunctionType) functionDef.name().typeV2()).returnType().unwrappedType()).isEqualTo(patternType);
  }

  @Test
  void missingTypePropagationWhenBuiltinIsReassigned() {
    FileInput fileInput = inferTypes(
      """
        builtin_str = str
        builtin_str
        str
        """
    );
    PythonType builtinStrType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    PythonType strType = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(builtinStrType).isEqualTo(STR_TYPE);
    assertThat(strType).isEqualTo(STR_TYPE);

    // SONARPY-2312: the reassigned "str = str" prevents the propagation of the built-in class type
    fileInput = inferTypes(
      """
        builtin_str = str
        str = str
        builtin_str
        str
        """
    );
    builtinStrType = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    strType = ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0).typeV2();
    assertThat(builtinStrType).isInstanceOf(UnknownType.class);
    assertThat(strType).isInstanceOf(UnknownType.class);
  }

  @Test
  void inferFunctionParameterTypes5() {
    FileInput root = inferTypes("""
      my_alias = int
      def foo(param: my_alias): ...
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(1);
    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void inferFunctionParameterTypes6() {
    FileInput root = inferTypes("""
      my_alias = int
      my_alias = str
      def foo(param: my_alias): ...
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(2);
    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferFunctionParameterTypes7() {
    FileInput root = inferTypes("""
      a = int
      def foo(param: a): ...
      """);
    var functionDef = (FunctionDef) root.statements().statements().get(1);
    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void inferFunctionParameterTypes8() {
    FileInput root = inferTypes("""
      if cond:
          my_alias = int
      else:
          my_alias = str
      def foo(param: my_alias): ...
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(root.statements().statements().size() - 1);
    assertThat(((FunctionType) functionDef.name().typeV2()).parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void inferFunctionParameterTypesMultiFile() {
    FileInput tree = parseWithoutSymbols(
      "def foo(param1: int): ...",
      "class A: ...",
      "def foo2(p1: dict, p2: A): ..."
    );
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    var modFile = pythonFile("mod.py");
    projectLevelSymbolTable.addModule(tree, "", modFile);
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    var modFileId = SymbolUtils.pathOf(modFile).toString();

    var intType = projectLevelTypeTable.lazyTypesContext().getOrCreateLazyType("int").resolve();
    var dictType = projectLevelTypeTable.lazyTypesContext().getOrCreateLazyType("dict").resolve();
    var aType = projectLevelTypeTable.lazyTypesContext().getOrCreateLazyType("mod.A").resolve();
    var lines = """
      from mod import foo, foo2
      foo
      foo2
      """;
    FileInput fileInput = inferTypes(lines, projectLevelTypeTable);
    FunctionType fooType = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(fooType.parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(intType);

    FunctionType foo2Type = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(foo2Type.parameters()).extracting(ParameterV2::declaredType).extracting(TypeWrapper::type).extracting(PythonType::unwrappedType).containsExactly(dictType, aType);
    assertThat(foo2Type.parameters()).extracting(ParameterV2::location).containsExactly(
      new LocationInFile(modFileId, 3, 9, 3, 17),
      new LocationInFile(modFileId, 3, 19, 3, 24));
  }

  @Test
  void inferFunctionReturnTypeType() {
    FileInput root = inferTypes("""
      from collections import namedtuple
      namedtuple
      """);

    var expr = (ExpressionStatement) root.statements().statements().get(root.statements().statements().size() - 1);
    assertThat(expr.expressions().get(0).typeV2()).isInstanceOf(FunctionType.class);
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
  void global_variable_in_nested_function() {
    FileInput fileInput = inferTypes("""
      def outer():
        a = 24
        def nested():
          global a
        a
      """);
    var outerFunctionDef = (FunctionDef) fileInput.statements().statements().get(0);
    SymbolV2 symbolV2 = ((Name) ((ExpressionStatement) outerFunctionDef.body().statements().get(2)).expressions().get(0)).symbolV2();
    assertThat(symbolV2.usages()).extracting(UsageV2::kind).containsExactlyInAnyOrder(UsageV2.Kind.ASSIGNMENT_LHS, UsageV2.Kind.GLOBAL_DECLARATION, UsageV2.Kind.OTHER);
  }

  @Test
  void nonlocal_variable_in_nested_function() {
    FileInput fileInput = inferTypes("""
      def outer():
        a = 24
        def nested():
          nonlocal a
        a
      """);
    var outerFunctionDef = (FunctionDef) fileInput.statements().statements().get(0);
    SymbolV2 symbolV2 = ((Name) ((ExpressionStatement) outerFunctionDef.body().statements().get(2)).expressions().get(0)).symbolV2();
    assertThat(symbolV2.usages()).extracting(UsageV2::kind).containsExactlyInAnyOrder(UsageV2.Kind.ASSIGNMENT_LHS, UsageV2.Kind.NONLOCAL_DECLARATION, UsageV2.Kind.OTHER);
  }

  @Test
  void nonlocal_variable_try_except() {
    FileInput fileInput = inferTypes("""
      def outer():
          contains_target = True
            
          def nested(item):
              nonlocal contains_target
              if cond():
                  contains_target = contains_target and foo()
            
          try:
              return contains_target
          except Exception as e:
              ...
          contains_target
      """);
    var outerFunctionDef = (FunctionDef) fileInput.statements().statements().get(0);
    SymbolV2 symbolV2 = ((Name) ((ExpressionStatement) outerFunctionDef.body().statements().get(3)).expressions().get(0)).symbolV2();
    assertThat(symbolV2.usages()).extracting(UsageV2::kind).containsExactlyInAnyOrder(
      UsageV2.Kind.ASSIGNMENT_LHS, UsageV2.Kind.NONLOCAL_DECLARATION, UsageV2.Kind.ASSIGNMENT_LHS, UsageV2.Kind.OTHER, UsageV2.Kind.OTHER, UsageV2.Kind.OTHER
    );
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
    var lastExpressionStatement = (ExpressionStatement) functionDef.body().statements().get(functionDef.body().statements().size() - 1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2()).isInstanceOf(UnionType.class);
    assertThat(((UnionType) lastExpressionStatement.expressions().get(0).typeV2()).candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE,
      STR_TYPE);
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

    assertThat(aName)
      .isNotNull()
      .extracting(Name::symbolV2)
      .isNotNull();

    var aSymbol = aName.symbolV2();
    assertThat(aSymbol.usages()).hasSize(6);

    var types = aSymbol.usages()
      .stream()
      .map(UsageV2::tree)
      .map(Name.class::cast)
      .sorted(Comparator.comparing(n -> n.firstToken().line()))
      .map(Expression::typeV2)
      .map(PythonType::unwrappedType)
      .toList();

    assertThat(types).hasSize(6)
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

    assertThat(aName)
      .isNotNull()
      .extracting(Name::symbolV2)
      .isNotNull();

    var aSymbol = aName.symbolV2();
    assertThat(aSymbol.usages()).hasSize(3);

    var types = aSymbol.usages()
      .stream()
      .map(UsageV2::tree)
      .map(Name.class::cast)
      .sorted(Comparator.comparing(n -> n.firstToken().line()))
      .map(Expression::typeV2)
      .toList();

    assertThat(types).hasSize(3);

    assertThat(types.get(0)).isInstanceOf(ObjectType.class)
      .extracting(ObjectType.class::cast)
      .extracting(ObjectType::type)
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("int");

    assertThat(types.get(1)).isInstanceOf(ObjectType.class)
      .extracting(ObjectType.class::cast)
      .extracting(ObjectType::type)
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("str");

    assertThat(types.get(2)).isInstanceOf(UnionType.class);
    var type3 = (UnionType) types.get(2);
    assertThat(type3.candidates())
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
    assertThat(lastExpression("""
      def f() -> int: pass
      f()
      """).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(lastExpression("""
      try: # try-catch activates the AstBasedTypeInference
        pass
      except:
        pass
      class A:
        def bar(self) -> int: pass
      def f() -> A: pass
      f().bar()
      """).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(lastExpression(
      """
        class A: pass
        A()
        """).typeV2().displayName()).contains("A");
  }

  @Test
  void primitive_variable_outside_function() {
    assertThat(lastExpression("a = 42; a").typeV2().unwrappedType()).isEqualTo(INT_TYPE);
  }

  static Stream<Arguments> primitiveVariableOutsideFunctionNotPropagatedTestSource() {
    return Stream.of(
      Arguments.of("a = 42"),
      Arguments.of("a = 3.14"),
      Arguments.of("a = 'hello'"),
      Arguments.of("a = True"),
      Arguments.of("a = None"),
      Arguments.of("a = 1j"),
      Arguments.of("a = bytes(1)"),
      Arguments.of("a = b'hello'"));
  }

  @MethodSource("primitiveVariableOutsideFunctionNotPropagatedTestSource")
  @ParameterizedTest
  void primitive_variable_outside_function_not_propagated(String code) {
    var fileInput = inferTypes(
      """
        %s
        def foo(): a
        """.formatted(code));

    // check initial assignment is not UNKNOWN
    var assignment = TreeUtils.firstChild(fileInput, AssignmentStatement.class::isInstance);
    assertThat(assignment)
      .isPresent()
      .map(AssignmentStatement.class::cast)
      .map(smnt -> smnt.assignedValue().typeV2())
      .containsInstanceOf(ObjectType.class);

    // check propagated value is UNKNOWN
    var optionalFoo = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance);
    assertThat(optionalFoo).isPresent();
    var foo = (FunctionDef) optionalFoo.get();
    var aName = TreeUtils.firstChild(foo.body(), tree -> tree instanceof Name name && "a".equals(name.name()));

    assertThat(aName)
      .isPresent()
      .map(Name.class::cast)
      .map(Name::typeV2)
      .contains(PythonType.UNKNOWN);
  }

  @Test
  void primitive_variable_outside_function_3() {
    assertThat(lastExpression(
      """
        def foo():
          a = 42
        a
        """).typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void primitive_variable_outside_function_4() {
    assertThat(lastExpression(
      """
        a = 42
        def foo():
          a = 'hello'
        a
        """).typeV2().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void class_instance_outside_function_5() {
    assertThat(lastExpression(
      """
        class A: ...
        a = A()
        def foo():
          a
        """).typeV2())
        .isInstanceOf(ObjectType.class)
        .extracting(PythonType::unwrappedType)
        .isInstanceOf(ClassType.class);
  }

  @Test
  void constant_outside_function() {
    var expr = lastExpression(
      """
        PI = 3.14
        def foo():
          PI
        """);
    assertThat(expr.typeV2().unwrappedType()).isEqualTo(FLOAT_TYPE);
  }

  @Test
  void typeVar_outside_function() {
    var expr = lastExpression(
      """
        from typing import TypeVar
        _T = TypeVar("_T")
        def foo(function: _T):
          function
        """);
    assertThat(expr.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOf(ClassType.class);
  }

  @Test
  void constant_after_usage_function() {
    var fileInput = inferTypes(
      """
        def foo():
          PI
        PI = 3.14
        """);

    var expr = TreeUtils.firstChild(fileInput, tree -> tree instanceof Name name && "PI".equals(name.name()));
    assertThat(expr)
      .map(Name.class::cast)
      .map(Name::typeV2)
      .hasValueSatisfying(name -> {
        assertThat(name)
          .isInstanceOf(ObjectType.class)
          .extracting(PythonType::unwrappedType)
          .isEqualTo(FLOAT_TYPE);
      });
  }

  @Test
  void constant_after_usage_lambda() {
    var fileInput = inferTypes(
      """
        stored_lambda = lambda: PI
        PI = 3.14
        """);

    var piExprInLambda = TreeUtils.firstChild(fileInput, tree -> tree instanceof Name name && "PI".equals(name.name()));
    assertThat(piExprInLambda)
      .map(Name.class::cast)
      .map(Name::typeV2)
      .hasValue(PythonType.UNKNOWN);

    var storedLambda = TreeUtils.firstChild(fileInput, tree -> tree instanceof Name name && "stored_lambda".equals(name.name()));
    assertThat(storedLambda)
      .map(Name.class::cast)
      .map(Name::typeV2)
      .hasValue(PythonType.UNKNOWN);
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

    PythonType nonePythonType = ObjectType.fromType(NONE_TYPE);
    PythonType myClassPythonType = ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();

    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    for (CallExpression call : calls) {
      PythonType calleeType = call.callee().typeV2();
      assertThat(calleeType).isInstanceOf(UnionType.class);
      assertThat(calleeType.isCompatibleWith(nonePythonType)).isEqualTo(TriBool.UNKNOWN);
      assertThat(calleeType.isCompatibleWith(myClassPythonType)).isEqualTo(TriBool.UNKNOWN);
    }
  }

  @Test
  void try_except_method_parameters() {
    FileInput fileInput = inferTypes("""
      class MyClass:
        def method(self, x:int):
          try:
            pass
          except:
            pass
          self.method()
          x
      """);

    CallExpression selfMethodCall = PythonTestUtils.getFirstChild(fileInput, CallExpression.class::isInstance);
    Name xName = PythonTestUtils.getFirstChild(fileInput, tree -> tree instanceof Name name && "x".equals(name.name()));
    Name selfName = PythonTestUtils.getFirstChild(fileInput, tree -> tree instanceof Name name && "self".equals(name.name()));

    assertThat(selfMethodCall.callee().typeV2()).isInstanceOf(FunctionType.class);

    assertThat(selfName.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOf(SelfType.class);

    assertThat(xName.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(INT_TYPE);
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
  void ast_based_type_inference_when_try_except_with_qualified_expression() {
    FileInput fileInput = inferTypes("""
      class MyClass:
        def foo(self, x):
          try:
            pass
          except:
            pass

          if ...:
            x = set()

          x.discard(1)
      """);

    QualifiedExpression xDiscardCallee = PythonTestUtils.getFirstChild(fileInput,
      tree -> tree instanceof QualifiedExpression qualifiedExpression && "discard".equals(qualifiedExpression.name().name()));
    CallExpression discardCall = PythonTestUtils.getFirstChild(fileInput, tree -> tree instanceof CallExpression call && call.callee() == xDiscardCallee);
    assertThat(discardCall.callee().typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void flow_insensitive_when_try_except_with_qualified_expression() {
    FileInput fileInput = inferTypes("""
      class MyClass:
        def foo(self, x):
          if ...:
            x = set()

          x.discard(1)
      """);

    QualifiedExpression xDiscardCallee = PythonTestUtils.getFirstChild(fileInput,
      tree -> tree instanceof QualifiedExpression qualifiedExpression && "discard".equals(qualifiedExpression.name().name()));
    CallExpression discardCall = PythonTestUtils.getFirstChild(fileInput, tree -> tree instanceof CallExpression call && call.callee() == xDiscardCallee);
    assertThat(discardCall.callee().typeV2()).isEqualTo(PythonType.UNKNOWN);
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

    assertThat(qualifiedExpression)
      .isNotNull()
      .extracting(QualifiedExpression::typeV2)
      .isNotNull();

    var qualifiedExpressionType = qualifiedExpression.typeV2();
    assertThat(qualifiedExpressionType)
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

    assertThat(qualifiedExpression)
      .isNotNull()
      .extracting(QualifiedExpression::typeV2)
      .isNotNull();

    var qualifiedExpressionType = qualifiedExpression.typeV2();
    assertThat(qualifiedExpressionType)
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

    assertThat(qualifiedExpressionType).isSameAs(bType);
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

    assertThat(qualifiedExpressionType)
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

    assertThat(qualifiedExpression)
      .isNotNull()
      .extracting(QualifiedExpression::typeV2)
      .isNotNull();

    var builtinsAppendType = LIST_TYPE.resolveMember("append").get();

    var qualifierType = qualifiedExpression.qualifier().typeV2().unwrappedType();
    assertThat(qualifierType).isSameAs(LIST_TYPE);

    var qualifiedExpressionType = qualifiedExpression.typeV2();
    assertThat(qualifiedExpressionType)
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

    assertThat(qualifiedExpression)
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

    assertThat(iType).isInstanceOf(ObjectType.class)
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

    assertThat(iType).isInstanceOf(ObjectType.class)
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

    assertThat(iType).isInstanceOf(UnionType.class);
    var candidates = ((UnionType) iType).candidates();
    assertThat(candidates)
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

    assertThat(iType).isInstanceOf(ObjectType.class)
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

    assertThat(iType).isInstanceOf(UnionType.class);

    var candidates = ((UnionType) iType).candidates();
    assertThat(candidates)
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

    assertThat(iType).isInstanceOf(ObjectType.class)
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

    assertThat(iType).isInstanceOf(UnknownType.class)
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

    assertThat(iType).isInstanceOf(UnknownType.class)
      .isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void typing_aliases_are_resolved_to_builtin_equivalent() {
    var fileInput = inferTypes("""
      import typing
      a = typing.Tuple
      a
      b = typing.List
      b
      c = typing.Dict
      c
      d = typing.Set
      d
      e = typing.FrozenSet
      e
      f = typing.Type
      f
      """);
    var aExpr = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0);
    assertThat(aExpr.typeV2()).isEqualTo(TUPLE_TYPE);
    var bExpr = ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0);
    assertThat(bExpr.typeV2()).isEqualTo(LIST_TYPE);
    var cExpr = ((ExpressionStatement) fileInput.statements().statements().get(6)).expressions().get(0);
    assertThat(cExpr.typeV2()).isEqualTo(DICT_TYPE);
    var dExpr = ((ExpressionStatement) fileInput.statements().statements().get(8)).expressions().get(0);
    assertThat(dExpr.typeV2()).isEqualTo(SET_TYPE);
    var eExpr = ((ExpressionStatement) fileInput.statements().statements().get(10)).expressions().get(0);
    assertThat(eExpr.typeV2()).isEqualTo(FROZENSET_TYPE);
    var fExpr = ((ExpressionStatement) fileInput.statements().statements().get(12)).expressions().get(0);
    assertThat(fExpr.typeV2()).isEqualTo(TYPE_TYPE);
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

    assertThat(iType).isInstanceOf(UnknownType.class)
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

    assertThat(lType).isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("list");

    assertThat(lType)
      .extracting(ObjectType.class::cast)
      .extracting(ObjectType::attributes)
      .asInstanceOf(InstanceOfAssertFactories.LIST)
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

    assertThat(qualifiedExpression)
      .isNotNull()
      .extracting(QualifiedExpression::typeV2)
      .isNotNull();

    var qualifiedExpressionType = qualifiedExpression.typeV2();
    assertThat(qualifiedExpressionType)
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

    assertThat(xType).extracting(PythonType::unwrappedType).isSameAs(STR_TYPE);
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

    assertThat(childClassType.hasMetaClass()).isTrue();

    var aType = TreeUtils.firstChild(root.statements().statements().get(3), ExpressionStatement.class::isInstance)
      .map(ExpressionStatement.class::cast)
      .flatMap(expressionStatement -> TreeUtils.firstChild(expressionStatement, Name.class::isInstance))
      .map(Name.class::cast)
      .map(Expression::typeV2)
      .get();

    assertThat(aType)
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


    assertThat(paramType).isInstanceOf(UnionType.class);
    assertThat(paramType.candidates())
      .hasSize(2);

    var candidatesUnwrappedType = paramType.candidates().stream()
      .map(PythonType::unwrappedType)
      .toList();

    assertThat(candidatesUnwrappedType)
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

    assertThat(paramType).isSameAs(PythonType.UNKNOWN);
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
  void unary_expression() {
    Function<Expression, PythonType> exprToType = expr -> expr.typeV2().unwrappedType();

    assertThat(lastExpression("-1")).extracting(exprToType).isEqualTo(INT_TYPE);
    assertThat(lastExpression("+1")).extracting(exprToType).isEqualTo(INT_TYPE);
    assertThat(lastExpression("-1.0")).extracting(exprToType).isEqualTo(FLOAT_TYPE);
    assertThat(lastExpression("+1.0")).extracting(exprToType).isEqualTo(FLOAT_TYPE);
    assertThat(lastExpression("-True")).extracting(exprToType).isEqualTo(INT_TYPE);
    assertThat(lastExpression("+True")).extracting(exprToType).isEqualTo(INT_TYPE);
    assertThat(lastExpression("~1")).extracting(exprToType).isEqualTo(INT_TYPE);
    assertThat(lastExpression("~True")).extracting(exprToType).isEqualTo(INT_TYPE);
    assertThat(lastExpression("not True")).extracting(exprToType).isEqualTo(BOOL_TYPE);
    assertThat(lastExpression("not 1")).extracting(exprToType).isEqualTo(BOOL_TYPE);
  }

  static Stream<Arguments> unary_expression_of_variables() {
    return Stream.of(
      Arguments.of("x = 1; -x", INT_TYPE),
      Arguments.of("x = 1; -(-x)", INT_TYPE),
      Arguments.of("x = 1; +x", INT_TYPE),
      Arguments.of("x = True; -x", INT_TYPE),
      Arguments.of("x = True; +x", INT_TYPE),
      Arguments.of("x = True; ~x", INT_TYPE),
      Arguments.of("x = True; not x", BOOL_TYPE),
      Arguments.of("x = 1; not x", BOOL_TYPE),
      Arguments.of("x = 1; y = -x; -y", INT_TYPE),
      Arguments.of("x = 1; x = 1; y = -x; -y", INT_TYPE),
      Arguments.of("x = 1; y = 1; y = -x; -y", INT_TYPE),

      Arguments.of("""
        if someCond:
          x = 1
        else:
          x = True
        y = x
        -y
        """, INT_TYPE),

      Arguments.of("x = 1; -(x + 1)", INT_TYPE)
    );
  }

  @ParameterizedTest
  @MethodSource("unary_expression_of_variables")
  void unary_expression_of_variables(String code, PythonType expectedType) {
    assertThat(lastExpression(code).typeV2()).isInstanceOf(ObjectType.class).extracting(PythonType::unwrappedType).isEqualTo(expectedType);
  }

  @ParameterizedTest
  @MethodSource("unary_expression_of_variables")
  void unary_expression_of_variables_with_try_except(String code, PythonType expectedType) {
    var codeWithTryCatch = """
      try: pass
      except: pass
      """ + code;
    assertThat(lastExpression(codeWithTryCatch)).extracting(expr -> expr.typeV2().unwrappedType()).isEqualTo(expectedType);
  }

  @Test
  void imported_ambiguous_symbol() {
    FileInput fileInput = inferTypes("""
      from os.path import realpath
      realpath
      """);
    UnionType realpathType = (UnionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(realpathType.candidates()).allMatch(FunctionType.class::isInstance);
    assertThat(realpathType.candidates()).extracting(PythonType::name).containsExactly("realpath", "realpath", "realpath", "realpath");
    assertThat(realpathType.candidates())
      .map(FunctionType.class::cast)
      .extracting(FunctionType::returnType)
      .extracting(PythonType::unwrappedType)
      .containsExactly(PythonType.UNKNOWN, PythonType.UNKNOWN, PythonType.UNKNOWN, PythonType.UNKNOWN);
  }

  @Test
  void imported_ambiguous_symbol_try_except() {
    FileInput fileInput = inferTypes("""
      try:
          from os.path import realpath
          realpath
      except:
          ...
      realpath
      """);
    Expression acosExpr1 = TreeUtils.firstChild(fileInput.statements().statements().get(0), ExpressionStatement.class::isInstance)
      .map(ExpressionStatement.class::cast)
      .map(ExpressionStatement::expressions)
      .map(expressions -> expressions.get(0))
      .get();
    UnionType acosType1 = (UnionType) acosExpr1.typeV2();
    assertThat(acosType1.candidates()).allMatch(FunctionType.class::isInstance);
    assertThat(acosType1.candidates())
      .map(FunctionType.class::cast)
      .extracting(FunctionType::returnType)
      .extracting(PythonType::unwrappedType)
      .containsExactly(PythonType.UNKNOWN, PythonType.UNKNOWN, PythonType.UNKNOWN, PythonType.UNKNOWN);

    UnionType acosType2 = (UnionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(acosType2.candidates()).allMatch(FunctionType.class::isInstance);
    assertThat(acosType2.candidates())
      .map(FunctionType.class::cast)
      .extracting(FunctionType::returnType)
      .extracting(PythonType::unwrappedType)
      .containsExactly(PythonType.UNKNOWN, PythonType.UNKNOWN, PythonType.UNKNOWN, PythonType.UNKNOWN);
  }

  @Test
  void return_type_of_call_of_locally_defined_function() {
    var type = lastExpression("""
      def foo() -> int: ...
      x = foo()
      x
      """).typeV2();
    assertThat(type.unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(type.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
  }

  @Test
  void type_origin_of_stub_function() {
    FileInput fileInput = inferTypes("""
      len
      x = len([1,2])
      x
      """);

    FunctionType lenType = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0).typeV2();
    assertThat(lenType.typeOrigin()).isEqualTo(TypeOrigin.STUB);
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(xType.typeSource()).isEqualTo(TypeSource.EXACT);

  }

  @Test
  void type_origin_of_project_function() {
    FileInput tree = parseWithoutSymbols(
      "def foo() -> int: ..."
    );
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);

    PythonType intType = projectLevelTypeTable.getBuiltinsModule().resolveMember("int").get();
    var lines = """
      from mod import foo
      foo
      x = foo()
      x
      """;
    FileInput fileInput = inferTypes(lines, projectLevelTypeTable);
    FunctionType fooType = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(fooType.typeOrigin()).isEqualTo(TypeOrigin.LOCAL);
    PythonType xType = ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0).typeV2();
    assertThat(xType.unwrappedType()).isEqualTo(intType);
  }


  @Test
  void type_project_function_with_self_return_type() {
    FileInput tree = parseWithoutSymbols("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...
      """);
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);

    var lines = """
      from mod import A

      a = A()
      a
      result = a.foo()
      result
      """;
    FileInput fileInput = inferTypes(lines, projectLevelTypeTable);

    PythonType aType = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(aType).isInstanceOf(ObjectType.class);
    assertThat(aType.unwrappedType()).isInstanceOf(ClassType.class);
    ClassType classType = (ClassType) aType.unwrappedType();
    assertThat(classType.name()).isEqualTo("A");
    assertThat(classType.fullyQualifiedName()).isEqualTo("mod.A");

    CallExpression callExpression = PythonTestUtils.getFirstChild(fileInput, t -> t  instanceof CallExpression fooCall && fooCall.callee() instanceof QualifiedExpression);
    assertThat(callExpression.callee().typeV2())
      .isInstanceOfSatisfying(FunctionType.class,
        functionType -> assertThat(functionType.returnType())
          .isInstanceOf(ObjectType.class)
          .extracting(PythonType::unwrappedType)
          .isInstanceOf(SelfType.class)
          .extracting(SelfType.class::cast)
          .extracting(SelfType::innerType)
          .isEqualTo(classType)
          );
    PythonType resultType = ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0).typeV2();
    assertThat(resultType.unwrappedType()).isInstanceOf(ClassType.class);
    assertThat(resultType.unwrappedType()).isEqualTo(classType);
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

    assertThat(((ExpressionStatement) fileInput.statements().statements().get(6)).expressions().get(0).typeV2()).isInstanceOf(UnionType.class);
    UnionType unionType = (UnionType) ((ExpressionStatement) fileInput.statements().statements().get(6)).expressions().get(0).typeV2().unwrappedType();
    assertThat(unionType.candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(classA, classB);
  }

  @Test
  void return_type_of_call_expression_inconsistent() {
    FileInput fileInput = inferTypes(
      """
        foo()
        """
    );
    CallExpression callExpression = ((CallExpression) ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0));
    CallExpression callExpressionSpy = Mockito.spy(callExpression);

    // Inconsistent union type, should not happen
    UnionType unionType = Mockito.mock(UnionType.class);
    Mockito.when(unionType.candidates()).thenReturn(Set.of(PythonType.UNKNOWN));

    Name mock = Mockito.mock(Name.class);
    Mockito.when(mock.typeV2()).thenReturn(unionType);
    Mockito.doReturn(mock).when(callExpressionSpy).callee();

    assertThat(callExpressionSpy.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void return_type_of_call_expression_inconsistent_2() {
    FileInput fileInput = inferTypes(
      """
        foo()
        """
    );
    CallExpression callExpression = ((CallExpression) ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0));
    CallExpression callExpressionSpy = Mockito.spy(callExpression);

    // Inconsistent union type, should not happen
    UnionType unionType = Mockito.mock(UnionType.class);
    Mockito.when(unionType.candidates()).thenReturn(Set.of());

    Name mock = Mockito.mock(Name.class);
    Mockito.when(mock.typeV2()).thenReturn(unionType);
    Mockito.doReturn(mock).when(callExpressionSpy).callee();

    assertThat(callExpressionSpy.typeV2()).isEqualTo(PythonType.UNKNOWN);
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
  void lazyTypeOfSuperType() {
    FileInput fileInput = inferTypes("""
      class MyClass(Exception):
          ...
      MyClass
      """);
    ClassType myClassType = (ClassType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(myClassType.superClasses()).extracting(TypeWrapper::type).containsExactly(EXCEPTION_TYPE);
  }


  @Test
  void lazyTypeOfSuperType2() {
    FileInput fileInput = inferTypes("""
      class A: ...
      A
      class MyClass(A, Exception, int):
          ...
      MyClass
      """);
    ClassType aType = (ClassType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    ClassType myClassType = (ClassType) ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0).typeV2();
    assertThat(myClassType.superClasses()).extracting(TypeWrapper::type).containsExactly(aType, EXCEPTION_TYPE, INT_TYPE);
  }

  @Test
  void resolvedTypingLazyType() {
    FileInput fileInput = inferTypes("""
      import calendar
      calendar.Calendar.iterweekdays
      """);
    FunctionType functionType = ((FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    PythonType returnType = functionType.returnType();
    assertThat(returnType.unwrappedType()).isInstanceOf(ClassType.class);
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
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(empty);
    LazyTypesContext lazyTypesContext = projectLevelTypeTable.lazyTypesContext();
    assertThat(lazyTypesContext.resolveLazyType(new LazyType("unknown", lazyTypesContext))).isEqualTo(PythonType.UNKNOWN);
    assertThat(lazyTypesContext.resolveLazyType(new LazyType("unrelated.unknown", lazyTypesContext))).isEqualTo(PythonType.UNKNOWN);
    assertThat(lazyTypesContext.resolveLazyType(new LazyType("typing.unknown", lazyTypesContext))).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void resolveIncorrectLazyType2() {
    ProjectLevelSymbolTable empty = ProjectLevelSymbolTable.empty();

    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(empty);
    LazyTypesContext lazyTypesContext = projectLevelTypeTable.lazyTypesContext();
    SymbolsModuleTypeProvider symbolsModuleTypeProvider = new SymbolsModuleTypeProvider(empty, lazyTypesContext);
    ModuleType builtinModule = symbolsModuleTypeProvider.getRootModule();
    symbolsModuleTypeProvider.convertModuleType(List.of("typing"), builtinModule);

    ClassSymbol symbol = Mockito.mock(ClassSymbolImpl.class);
    Mockito.when(symbol.kind()).thenReturn(Symbol.Kind.OTHER);
    assertThat(PROJECT_LEVEL_TYPE_TABLE.lazyTypesContext().getOrCreateLazyType("typing.Iterable.unknown").resolve()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void convertTypeshedModuleWithAliases() {
    ProjectLevelSymbolTable empty = ProjectLevelSymbolTable.empty();
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(empty);
    LazyTypesContext lazyTypesContext = projectLevelTypeTable.lazyTypesContext();
    SymbolsModuleTypeProvider symbolsModuleTypeProvider = new SymbolsModuleTypeProvider(empty, lazyTypesContext);
    ModuleType builtinModule = symbolsModuleTypeProvider.getRootModule();
    PythonType responses = symbolsModuleTypeProvider.convertModuleType(List.of("fastapi", "responses"), builtinModule);
    assertThat(responses.resolveMember("FileResponse")).containsInstanceOf(ClassType.class);
    PythonType concurrency = symbolsModuleTypeProvider.convertModuleType(List.of("fastapi", "concurrency"), builtinModule);
    assertThat(concurrency.resolveMember("iterate_in_threadpool")).containsInstanceOf(FunctionType.class);

    List<Symbol> fileResponseSymbols = empty.stubFilesSymbols().stream().filter(s -> "fastapi.responses.FileResponse".equals(s.fullyQualifiedName())).toList();
    assertThat(fileResponseSymbols).hasSize(1);
    assertThat(fileResponseSymbols.get(0).kind()).isEqualTo(Symbol.Kind.CLASS);
    List<Symbol> runInThreadPoolSymbols = empty.stubFilesSymbols().stream().filter(s -> "fastapi.concurrency.run_in_threadpool".equals(s.fullyQualifiedName())).toList();
    assertThat(runInThreadPoolSymbols).hasSize(1);
    assertThat(runInThreadPoolSymbols.get(0).kind()).isEqualTo(Symbol.Kind.FUNCTION);
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
  void import_in_function() {
    Expression fcntlExpr = lastExpression("""
      def foo():
        import fcntl
      fcntl
      """);
    assertThat(fcntlExpr.typeV2()).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void basic_imported_symbols() {
    FileInput fileInput = inferTypes(
      """
        import fcntl, os.path
        fcntl
        os.path
        """
    );
    PythonType fnctlModule = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(fnctlModule).isInstanceOf(ModuleType.class);
    assertThat(fnctlModule.name()).isEqualTo("fcntl");
    PythonType osPathModule = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(osPathModule).isInstanceOf(ModuleType.class);
    assertThat(osPathModule.name()).isEqualTo("path");
    assertThat(((UnionType) osPathModule.resolveMember("realpath").get()).candidates()).allMatch(FunctionType.class::isInstance);
  }

  // TODO SONARPY-2176 ProjectLevelSymbolTable#getType should be able to resolve types when there is a conflict between a member and a subpackage
  @Test
  void import_conflict_between_member_and_submodule() {
    var statement = lastExpression("""
      import opentracing.tracer.Tracer as ottt
      ottt
      """);
    assertThat(((UnresolvedImportType) statement.typeV2())).extracting(UnresolvedImportType::importPath).isEqualTo("opentracing.tracer.Tracer");
  }

  @Test
  void import_unknown_httpx_client() {
    var statement = lastExpression("""
      import httpx
      httpx.Client()
      """);
    assertThat(statement.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void returnTypeOfTypeshedSymbol() {
    FileInput fileInput = inferTypes("""
      from sys import gettrace
      gettrace
      """);
    FunctionType functionType = ((FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    assertThat(functionType.fullyQualifiedName()).isEqualTo("sys.gettrace");
    PythonType returnType = functionType.returnType();
    UnionType unionType = (UnionType) returnType.unwrappedType();
    assertThat(unionType.candidates()).extracting(PythonType::name).containsExactlyInAnyOrder("function", "NoneType");
  }

  @Test
  void isInstanceTests() {
    var xType = lastExpression("""
      def foo(x: int):
        if isinstance(x, Foo):
          ...
        x
      """).typeV2();
    assertThat(xType).isSameAs(PythonType.UNKNOWN);

    xType = lastExpression("""
      def foo(x: int):
        if isinstance(x):
          ...
        x
      """).typeV2();
    assertThat(xType.unwrappedType()).isSameAs(INT_TYPE);

    xType = lastExpression("""
      def foo(x: int):
        if isinstance(x.b, Foo):
          ...
        x
      """).typeV2();
    assertThat(xType.unwrappedType()).isSameAs(INT_TYPE);

    xType = lastExpression("""
      def foo():
        x = 10
        if isinstance(x, Foo):
          ...
        x
      """).typeV2();
    assertThat(xType.unwrappedType()).isSameAs(INT_TYPE);

    xType = lastExpression("""
      def foo(x: list):
        if isinstance(**x, Foo):
          ...
        x
      """).typeV2();

    assertThat(xType.unwrappedType()).isSameAs(LIST_TYPE);
  }

  @Test
  void assignmentOrTest() {
    var yType = (UnionType) lastExpression("""
      def foo(x: int):
        y = x or "str"
        y
      """).typeV2();

    assertThat(yType.candidates())
      .allMatch(ObjectType.class::isInstance)
      .extracting(PythonType::unwrappedType)
      .containsOnly(INT_TYPE, STR_TYPE);
    assertThat(yType.typeSource()).isSameAs(TypeSource.TYPE_HINT);
  }

  public static Stream<Arguments> binaryExpressionsSource() {
    return Stream.of(
      Arguments.of("""
        1 + 2
        """, ObjectType.fromType(INT_TYPE)),
        Arguments.of("""
        1 - 2
        """, ObjectType.fromType(INT_TYPE)),
        Arguments.of("""
        1 * 2
        """, ObjectType.fromType(INT_TYPE)),
        Arguments.of("""
        1 / 2
        """, ObjectType.fromType(INT_TYPE)),
        Arguments.of("""
        '1' + '2'
        """, ObjectType.fromType(STR_TYPE)),
        Arguments.of("""
        1 + '2'
        """, PythonType.UNKNOWN),
      Arguments.of("""
        1 + 2.0
        """, PythonType.UNKNOWN),
      Arguments.of("""
        a = 1
        b = 2
        a + b
        """, ObjectType.fromType(INT_TYPE)),
        Arguments.of("""
        a = 1
        b = 2
        c = a - b
        c
        """, ObjectType.fromType(INT_TYPE)),
        Arguments.of("""
        try:
          ...
        except:
          ...
        a = 1
        b = 2
        c = a - b
        c
        """, ObjectType.fromType(INT_TYPE))
    );
  }

  @ParameterizedTest
  @MethodSource("binaryExpressionsSource")
  void binaryExpressionTest(String code, PythonType expectedType) {
    assertThat(lastExpression(code).typeV2()).isEqualTo(expectedType);
  }

  @Test
  void assignmentPlusTest() {
    var fileInput = inferTypes("""
      class A: ...
      def foo(x: int, y: str):
        a = x + 1
        b = x + y
        c = A + x
        d = x + A
        e = 2 + 3
        a
        b
        c
        d
        e
      """);

    var statements = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::body)
      .map(StatementList::statements)
      .orElseGet(List::of);

    var aType = ((ExpressionStatement) statements.get(statements.size() - 5)).expressions().get(0).typeV2();
    var bType = ((ExpressionStatement) statements.get(statements.size() - 4)).expressions().get(0).typeV2();
    var cType = ((ExpressionStatement) statements.get(statements.size() - 3)).expressions().get(0).typeV2();
    var dType = ((ExpressionStatement) statements.get(statements.size() - 2)).expressions().get(0).typeV2();
    var eType = ((ExpressionStatement) statements.get(statements.size() - 1)).expressions().get(0).typeV2();

    assertThat(aType.unwrappedType()).isSameAs(INT_TYPE);
    assertThat(aType.typeSource()).isSameAs(TypeSource.TYPE_HINT);

    assertThat(bType).isSameAs(PythonType.UNKNOWN);
    assertThat(cType).isSameAs(PythonType.UNKNOWN);
    assertThat(dType).isSameAs(PythonType.UNKNOWN);

    assertThat(eType.unwrappedType()).isSameAs(INT_TYPE);
    assertThat(eType.typeSource()).isSameAs(TypeSource.EXACT);
  }

  @Test
  void assignmentPlusTest2() {
    var fileInput = inferTypes("""
      class A: ...
      def foo(x):
        if x:
          t = int
        else:
          t = str
        a = t()
        b = 1 + a
        c = a + 1
        a
        b
        c
      """);

    var statements = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::body)
      .map(StatementList::statements)
      .orElseGet(List::of);

    var aType = ((ExpressionStatement) statements.get(statements.size() - 3)).expressions().get(0).typeV2();
    var bType = ((ExpressionStatement) statements.get(statements.size() - 2)).expressions().get(0).typeV2();
    var cType = ((ExpressionStatement) statements.get(statements.size() - 1)).expressions().get(0).typeV2();

    assertThat(aType).isInstanceOf(UnionType.class);
    assertThat(((UnionType) aType).candidates()).extracting(PythonType::unwrappedType).containsOnly(INT_TYPE, STR_TYPE);


    assertThat(bType).isSameAs(PythonType.UNKNOWN);
    assertThat(cType).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void assignmentMinusTest() {
    var fileInput = inferTypes("""
      def foo():
        f = 2 - 3
        f
      """);

    var statements = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::body)
      .map(StatementList::statements)
      .orElseGet(List::of);

    var fType = ((ExpressionStatement) statements.get(statements.size() - 1)).expressions().get(0).typeV2();
    assertThat(fType).isInstanceOf(ObjectType.class).extracting(PythonType::unwrappedType).isSameAs(INT_TYPE);
  }

  @Test
  void assignmentTemplateString() {
    var fileInput = inferTypes("""
      def template():
        a = t"test{name}"
        b = ""
        c = T"some {f'test{name}'}"
        a
        b
        c
      """);

    var statements = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::body)
      .map(StatementList::statements)
      .orElseGet(List::of);

    var aType = ((ExpressionStatement) statements.get(statements.size() - 3)).expressions().get(0).typeV2();
    var bType = ((ExpressionStatement) statements.get(statements.size() - 2)).expressions().get(0).typeV2();
    var cType = ((ExpressionStatement) statements.get(statements.size() - 1)).expressions().get(0).typeV2();
    assertThat(aType).extracting(PythonType::unwrappedType).isSameAs(PythonType.UNKNOWN);
    assertThat(bType).extracting(PythonType::unwrappedType).isEqualTo(STR_TYPE);
    assertThat(cType).extracting(PythonType::unwrappedType).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void assignmentStatementLhsTypeTest() {
    var fileInput = inferTypes("""
      def foo():
          subscription[call('goal_{}'.format(1))] = 1
      """);

    var literal = TreeUtils.firstChild(fileInput, StringLiteral.class::isInstance)
      .map(StringLiteral.class::cast)
      .get();

    var type = literal.typeV2();
    assertThat(type.unwrappedType()).isSameAs(STR_TYPE);
  }


  @Test
  void typesBySymbol_function_def() {
    var typesBySymbol = inferTypesBySymbol("""
      def foo():
        ...
      """);
    assertThat(typesBySymbol).hasSize(1);
    SymbolV2 symbolV2 = typesBySymbol.keySet().iterator().next();
    assertThat(symbolV2.name()).isEqualTo("foo");
    Set<PythonType> types = typesBySymbol.get(symbolV2);
    assertThat(types).hasSize(1);
    PythonType type = types.iterator().next();
    assertThat(type).isInstanceOf(FunctionType.class).extracting(PythonType::name).isEqualTo("foo");
  }

  @Test
  void moduleType_function_def() {
    var moduleType = inferModuleType("""
      def foo():
        ...
      """);
    assertThat(moduleType.members()).hasSize(1);
  }

  @Test
  void typesBySymbol_class_def() {
    var typesBySymbol = inferTypesBySymbol("""
      class A:
        ...
      """);
    assertThat(typesBySymbol).hasSize(1);
    SymbolV2 symbolV2 = typesBySymbol.keySet().iterator().next();
    assertThat(symbolV2.name()).isEqualTo("A");
    Set<PythonType> types = typesBySymbol.get(symbolV2);
    assertThat(types).hasSize(1);
    PythonType type = types.iterator().next();
    assertThat(type).isInstanceOf(ClassType.class).extracting(PythonType::name).isEqualTo("A");
  }

  @Test
  void typesBySymbol_variable() {
    var typesBySymbol = inferTypesBySymbol("""
      a = 10
      """);
    assertThat(typesBySymbol).hasSize(1);
    SymbolV2 symbolV2 = typesBySymbol.keySet().iterator().next();
    assertThat(symbolV2.name()).isEqualTo("a");
    Set<PythonType> types = typesBySymbol.get(symbolV2);
    assertThat(types).hasSize(1);
    PythonType type = types.iterator().next();
    assertThat(type).isInstanceOf(ObjectType.class).extracting(PythonType::unwrappedType).extracting(PythonType::name).isEqualTo("int");
  }

  @Test
  void typesBySymbol_reassigned_variable() {
    var typesBySymbol = inferTypesBySymbol("""
      a = 10
      if b:
        a = "hello"
      """);
    assertThat(typesBySymbol).hasSize(1);
    SymbolV2 symbolV2 = typesBySymbol.keySet().iterator().next();
    assertThat(symbolV2.name()).isEqualTo("a");
    Set<PythonType> types = typesBySymbol.get(symbolV2);
    assertThat(types).hasSize(2);
    assertThat(types).extracting(Object::getClass).extracting(Class.class::cast).containsOnly(ObjectType.class);
    assertThat(types).extracting(PythonType::unwrappedType).extracting(PythonType::name).containsExactlyInAnyOrder("int", "str");
  }

  @Test
  void typesBySymbol_class_def_overwrite_imported_type() {
    var typesBySymbol = inferTypesBySymbol("""
      from something import A
      if b:
        class A: ...
      """);
    assertThat(typesBySymbol).hasSize(1);
    SymbolV2 symbolV2 = typesBySymbol.keySet().iterator().next();
    assertThat(symbolV2.name()).isEqualTo("A");
    Set<PythonType> types = typesBySymbol.get(symbolV2);
    assertThat(types).hasSize(2);
    assertThat(types).extracting(Object::getClass).extracting(Class.class::cast).containsOnly(ClassType.class, UnresolvedImportType.class);
  }

  @Test
  void typesBySymbol_try_except() {
    var typesBySymbol = inferTypesBySymbol("""
      from something import A
      A = 10
      try:
        class A: ...
      except:
        def A():
          b = 10
      """);
    assertThat(typesBySymbol).hasSize(1);
    SymbolV2 symbolV2 = typesBySymbol.keySet().iterator().next();
    assertThat(symbolV2.name()).isEqualTo("A");
    Set<PythonType> types = typesBySymbol.get(symbolV2);
    assertThat(types).hasSize(4);
    assertThat(types).extracting(Object::getClass).extracting(Class.class::cast).containsOnly(UnresolvedImportType.class, ClassType.class, FunctionType.class, ObjectType.class);
  }

  @Test
  void typesBySymbol_declaration_without_assignment() {
    // SONARPY-2218 variable declaration without assignment is not supported
    var typesBySymbol = inferTypesBySymbol("""
      A : int
      A = None
      """);
    assertThat(typesBySymbol).isEmpty();
  }

  @Test
  void typeBySymbol_invalidCfg() {
    // No types will be retrieved when the CFG is invalid
    var typesBySymbol = inferTypesBySymbol("""
      class A: ...
      continue
      """);
    assertThat(typesBySymbol).isEmpty();
  }

  @Test
  void functionTypeUnknownDecoratorsTest() {
    var fileInput = inferTypes("""
      @unknown
      def foo(): ...
      """);

    var fooName = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::name)
      .get();

    var type = fooName.typeV2();
    assertThat(type).isInstanceOfSatisfying(FunctionType.class, functionType -> {
      assertThat(functionType.decorators()).hasSize(1)
        .extracting(TypeWrapper::type)
        .containsOnly(PythonType.UNKNOWN);
    });
  }

  @Test
  void functionTypeLocallyDefinedDecoratorTest() {
    var fileInput = inferTypes("""
      def known_decorator(): ...
      @known_decorator
      def foo(): ...
      """);

    var knownDecoratorName = TreeUtils.firstChild(fileInput.statements().statements().get(0), FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::name)
      .get();

    var knownDecoratorType = knownDecoratorName.typeV2();

    var fooName = TreeUtils.firstChild(fileInput.statements().statements().get(1), FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::name)
      .get();

    var fooType = fooName.typeV2();
    assertThat(fooType).isInstanceOfSatisfying(FunctionType.class, functionType -> {
      assertThat(functionType.decorators()).hasSize(1)
        .extracting(TypeWrapper::type)
        .containsOnly(knownDecoratorType);
    });
  }

  @Test
  void functionTypeQualifiedExpressionDecoratorTest() {
    var fileInput = inferTypes("""
      import lib.unknown
      @lib.unknown.known_decorator
      def foo(): ...
      """);

    var fooName = TreeUtils.firstChild(fileInput.statements().statements().get(1), FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::name)
      .get();

    var fooType = fooName.typeV2();
    assertThat(fooType).isInstanceOfSatisfying(FunctionType.class, functionType -> assertThat(functionType.decorators()).hasSize(1)
      .extracting(TypeWrapper::type)
      .extracting(UnresolvedImportType.class::cast)
      .extracting(UnresolvedImportType::importPath)
      .containsOnly("lib.unknown.known_decorator"));
  }

  @Test
  @Disabled("SONARPY-2248")
  void typesBySymbol_global_statement() {
    var typesBySymbol = inferTypesBySymbol("""
      class C:
        pass
      global C
      """);
    assertThat(typesBySymbol).isNotEmpty();
    assertThat(typesBySymbol.values().iterator().next()).isInstanceOf(ClassType.class);
  }

  @Test
  void wildCardImportsMultiFile() {
    FileInput tree = parseWithoutSymbols("""
      def foo(): pass
      def bar(): pass
      """);
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    var modFile = pythonFile("mod.py");
    projectLevelSymbolTable.addModule(tree, "", modFile);
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);

    var lines = """
      from mod import *
      from mod import foo as imported_foo
      (foo, imported_foo)
      """;
    TupleImpl tupleExpr = (TupleImpl) lastExpression(lines, projectLevelTypeTable);
    List<Expression> tupleExpressions = tupleExpr.elements();
    Expression fooExpr = tupleExpressions.get(0);
    Expression importedFooExpr = tupleExpressions.get(1);

    assertThat(fooExpr.typeV2()).isInstanceOf(FunctionType.class);
    assertThat(importedFooExpr.typeV2()).isInstanceOf(FunctionType.class);
  }

  @Test
  void conflictingWildCardImportsMultiFile() {
    TupleImpl tupleExpr = (TupleImpl) new TestProject()
      .addModule("mod1.py", "def foo(): pass")
      .addModule("mod2.py", "def foo(): pass")
      .lastExpression("""
        from mod1 import *
        from mod2 import *
        from mod1 import foo as incorrect_foo
        from mod2 import foo as correct_foo
        (foo, incorrect_foo, correct_foo)
        """);

    List<Expression> tupleExpressions = tupleExpr.elements();
    Expression fooExpr = tupleExpressions.get(0);
    Expression incorrectFoo = tupleExpressions.get(1);
    Expression correctFoo = tupleExpressions.get(2);

    assertThat(fooExpr.typeV2())
      .isInstanceOf(FunctionType.class)
      .isEqualTo(correctFoo.typeV2())
      .isNotEqualTo(incorrectFoo.typeV2());
  }

  public static Stream<Arguments> wildCardImportConflictsWithOtherImportMultiFileSource() {
    return Stream.of(
      Arguments.argumentSet("wildcard first", """
        from mod1 import *
        from mod2 import foo
        import mod1
        import mod2
        (foo, mod1.foo, mod2.foo)
        """),
      // foo should be mod2.foo, not mod1.foo. Since wildcard imports are only assigned to names without a symbol, foo will be equal to mod1.foo. See SONARPY-2357
      Arguments.argumentSet("wildcard second", """
        from mod1 import foo
        from mod2 import *
        import mod1
        import mod2
        (foo, mod2.foo, mod1.foo)
        """)
    );
  }

  @MethodSource("wildCardImportConflictsWithOtherImportMultiFileSource")
  @ParameterizedTest
  void wildCardImportConflictsWithOtherImportMultiFile(String sourcecode) {
    TupleImpl tupleExpr = (TupleImpl) new TestProject()
      .addModule("mod1.py", "def foo(): pass")
      .addModule("mod2.py", "def foo(): pass")
      .lastExpression(sourcecode);

    List<Expression> tupleExpressions = tupleExpr.elements();
    Expression fooExpr = tupleExpressions.get(0);
    Expression incorrectFoo = tupleExpressions.get(1);
    Expression correctFoo = tupleExpressions.get(2);

    assertThat(fooExpr.typeV2())
      .isInstanceOf(FunctionType.class)
      .isEqualTo(correctFoo.typeV2())
      .isNotEqualTo(incorrectFoo.typeV2());
  }

  @Test
  void wildCardInInnerScopeMultiFile() {
    Expression fooExpr = new TestProject()
      .addModule("mod.py", "def foo(): pass")
      .lastExpression("""
        def bar():
          from mod import *
        foo
        """);

    assertThat(fooExpr.typeV2()).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void wildCardInInnerScopeMultiFileWithSameFunctionDefinitionConflict() {
    FileInput fileInput = new TestProject()
      .addModule("mod.py", "def foo(): pass")
      .inferTypes("""
        def bar():
          from mod import *
          foo
        def foo(): pass
        """);

    FunctionDef barFunctionDef = PythonTestUtils.getFirstDescendant(fileInput, tree -> tree instanceof FunctionDef functionDef && "bar".equals(functionDef.name().name()));
    FunctionDef fooFunctionDef = PythonTestUtils.getFirstDescendant(fileInput, tree -> tree instanceof FunctionDef functionDef && "foo".equals(functionDef.name().name()));
    Name fooExpr = PythonTestUtils.getFirstDescendant(barFunctionDef.body(), tree -> tree instanceof Name name && "foo".equals(name.name()));
    // fooExpr should be mod.foo, but is resolved to foo because wildcard imports are only applied to names without a symbol. See SONARPY-2357
    assertThat(fooExpr.typeV2()).isEqualTo(fooFunctionDef.name().typeV2());
  }

  @Test
  void wildCardAfterUseMultiFile() {
    Expression vExpr = new TestProject()
      .addModule("mod.py", "def foo(): pass")
      .lastExpression("""
        v = foo
        from mod import *
        v
        """);

    assertThat(vExpr.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void reexportedWildcardImport() {
    TupleImpl tupleExpr = new TestProject()
      .addModule("mod1.py", "def foo(): pass")
      .addModule("mod2.py", "from mod1 import *")
      .lastExpressionAsTuple("""
        import mod2
        from mod1 import foo as actual_foo
        (mod2.foo, actual_foo)
        """);

    var importedFoo = tupleExpr.elements().get(0);
    var actualFoo = tupleExpr.elements().get(1);

    assertThat(importedFoo.typeV2())
      .isEqualTo(PythonType.UNKNOWN)
      .isNotEqualTo(actualFoo.typeV2());
  }

  @Test
  void pysparkStubs() {
    var fileInput = inferTypes("""
      from pyspark.sql import SparkSession
      SparkSession
      """);
    var sparkSessionType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    ClassType classType = (ClassType) sparkSessionType;
    assertThat(classType.fullyQualifiedName()).isEqualTo("pyspark.sql.session.SparkSession");
    
    Member nestedBuilderMember = classType.members().stream().filter(m -> m.name().equals("Builder")).findFirst().get();
    assertThat(nestedBuilderMember.type()).isInstanceOf(ClassType.class);
    assertThat(((ClassType) nestedBuilderMember.type()).fullyQualifiedName()).isEqualTo("pyspark.sql.session.SparkSession.Builder");
  }

  @Test
  void propertyMemberAccess() {
    FileInput fileInput = inferTypes("""
      class A:
        @property
        def foo(self) -> int:
          return 42
      a = A()
      a.foo
      A.foo
      """);
    var qualifiedExpression1 = ((QualifiedExpression) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0));
    var qualifiedExpression2 = ((QualifiedExpression) ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0));
    assertThat(qualifiedExpression1.name().typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(qualifiedExpression2.name().typeV2().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void customPropertyMemberAccess() {
    FileInput fileInput = inferTypes("""
      class CustomProperty(property): ...
      class A:
        @CustomProperty
        def foo(self) -> int:
          return 42
      a = A()
      a.foo
      A.foo
      """);
    var qualifiedExpression1 = ((QualifiedExpression) ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0));
    var qualifiedExpression2 = ((QualifiedExpression) ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0));
    assertThat(qualifiedExpression1.name().typeV2().unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(qualifiedExpression2.name().typeV2().unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void importFromModulesType() {
    var rootModule = new TestProject()
      .addModule("f1", "def foo(): ...");

    var fileInput = rootModule.inferTypes("from f1 import foo");
    var importFrom = (ImportFrom) fileInput.statements().statements().get(0);
    var types = importFrom.module().names().stream().map(Name::typeV2).toList();
    assertThat(types.get(0)).isInstanceOfSatisfying(ModuleType.class, moduleType -> assertThat(moduleType.fullyQualifiedName()).isEqualTo("f1"));

  }

  @Test
  void userDefinedEnumFieldType() {
    var root = inferTypes("""
      from enum import Enum
      class Color(Enum):
        RED = 1

      Color.RED
      """);

    var qualifiedExpr = (QualifiedExpression) TreeUtils.firstChild(root, QualifiedExpression.class::isInstance).get();

    assertThat(qualifiedExpr.typeV2())
      .isSameAs(PythonType.UNKNOWN); // TODO SONARPY-3282: should be ObjectType[Color]
  }

  @Test
  void typeshedDefinedEnumFieldType() {
    var socketSockRawExpr = lastExpression("""
      from socket import SocketKind
      SocketKind.SOCK_RAW
      """);

    assertThat(socketSockRawExpr.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isSameAs(INT_TYPE); // TODO SONARPY-3283: should be ObjectType[SocketKind]
  }

  @Test
  void importResolutionWithRelativeImportInInitFileTest() {
    // Reproduce case SONARPY-3470: Relative import in __init__.py should resolve correctly (account for the fact module FQN already truncates __init__.py)
    var project = new TestProject();
    project.addModule("anchore_engine/__init__.py", "");
    project.addModule("anchore_engine/db/__init__.py", "");
    project.addModule("anchore_engine/db/entities/__init__.py", "");
    project.addModule("anchore_engine/db/entities/common.py", """
      def session_scope(): ...
      """);

    FileInput dbInitFileInput = project.inferTypes("anchore_engine/db/__init__.py", """
      from .entities.common import session_scope
      session_scope
      """);

    var statements = dbInitFileInput.statements().statements();
    var sessionScopeType = ((ExpressionStatement) statements.get(1)).expressions().get(0).typeV2();

    var sessionScopeChecker = project.typeCheckBuilder().isTypeOrInstanceWithName("anchore_engine.db.entities.common.session_scope");

    assertThat(sessionScopeChecker.check(sessionScopeType)).isEqualTo(TriBool.TRUE);
  }

  private static Map<SymbolV2, Set<PythonType>> inferTypesBySymbol(String lines) {
    FileInput root = parse(lines);
    var symbolTable = new SymbolTableBuilderV2(root).build();
    var typeInferenceV2 = new TypeInferenceV2(PROJECT_LEVEL_TYPE_TABLE, pythonFile, symbolTable, "");
    return typeInferenceV2.inferTypes(root);
  }

  private static ModuleType inferModuleType(String lines) {
    FileInput root = parse(lines);
    var symbolTable = new SymbolTableBuilderV2(root).build();
    var typeInferenceV2 = new TypeInferenceV2(PROJECT_LEVEL_TYPE_TABLE, pythonFile, symbolTable, "");
    return typeInferenceV2.inferModuleType(root);
  }

  private static FileInput inferTypes(String lines) {
    return inferTypes(lines, PROJECT_LEVEL_TYPE_TABLE);
  }

  public static FileInput inferTypes(String lines, ProjectLevelTypeTable projectLevelTypeTable) {
    FileInput root = parse(lines);

    var symbolTable = new SymbolTableBuilderV2(root)
      .build();
    new TypeInferenceV2(projectLevelTypeTable, pythonFile, symbolTable, "my_package").inferTypes(root);
    return root;
  }

  public static Expression lastExpression(String lines) {
    return lastExpression(lines, PROJECT_LEVEL_TYPE_TABLE);
  }

  public static Expression lastExpression(String lines, ProjectLevelTypeTable projectLevelTypeTable) {
    FileInput fileInput = inferTypes(lines, projectLevelTypeTable);
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

  @Test
  void selfParameterTypeInMethod() {
    FileInput root = inferTypes("""
      class MyClass:
        def foo(self, x):
          self
        def bar(self):
          ...
      """);

    var classDef = (ClassDef) root.statements().statements().get(0);
    var fooMethodDef = (FunctionDef) classDef.body().statements().get(0);
    var selfExpressionStatement = (ExpressionStatement) fooMethodDef.body().statements().get(0);
    var selfExpression = selfExpressionStatement.expressions().get(0);
    var selfType = selfExpression.typeV2();

    // The self parameter should have type ObjectType[SelfType[MyClass]]
    assertThat(selfType).isInstanceOf(ObjectType.class);
    var objectType = (ObjectType) selfType;
    assertThat(objectType.unwrappedType()).isInstanceOf(SelfType.class);
    
    var selfTypeInner = (SelfType) objectType.unwrappedType();
    var myClassType = classDef.name().typeV2();
    assertThat(selfTypeInner.innerType()).isEqualTo(myClassType);
  }

  @Test
  void selfParameterTypeInMethodWithNonStandardName() {
    FileInput root = inferTypes("""
      class MyClass:
        def foo(this):
          this
      """);

    var classDef = (ClassDef) root.statements().statements().get(0);
    var fooMethodDef = (FunctionDef) classDef.body().statements().get(0);
    var thisExpressionStatement = (ExpressionStatement) fooMethodDef.body().statements().get(0);
    var thisExpression = thisExpressionStatement.expressions().get(0);

    assertThat(thisExpression.typeV2()).isInstanceOf(ObjectType.class);
    var objectType = (ObjectType) thisExpression.typeV2();
    assertThat(objectType.unwrappedType()).isInstanceOf(SelfType.class);
  }

  @Test
  void selfParameterTypeOverwritesExplicitAnnotation() {
    FileInput root = inferTypes("""
      class MyClass:
        def foo(self: int):
          self
      """);

    var classDef = (ClassDef) root.statements().statements().get(0);
    var fooMethodDef = (FunctionDef) classDef.body().statements().get(0);
    var selfExpressionStatement = (ExpressionStatement) fooMethodDef.body().statements().get(0);
    var selfExpression = selfExpressionStatement.expressions().get(0);

    // SelfType overwrites the explicit type annotation
    assertThat(selfExpression.typeV2()).isInstanceOf(ObjectType.class);
    var objectType = (ObjectType) selfExpression.typeV2();
    assertThat(objectType.unwrappedType()).isInstanceOf(SelfType.class);
  }

  @Test
  void classMethodFirstParameterType() {
    FileInput root = inferTypes("""
      class MyClass:
        @classmethod
        def foo(cls):
          cls
      """);

    var classDef = (ClassDef) root.statements().statements().get(0);
    var fooMethodDef = (FunctionDef) classDef.body().statements().get(0);
    var clsExpressionStatement = (ExpressionStatement) fooMethodDef.body().statements().get(0);
    var clsExpression = clsExpressionStatement.expressions().get(0);

    assertThat(clsExpression.typeV2())
      .isInstanceOfSatisfying(SelfType.class, selfType -> assertThat(selfType.innerType())
        .isEqualTo(classDef.name().typeV2()));
  }

  @Test
  void staticMethodHasNoImplicitSelfParameter() {
    FileInput root = inferTypes("""
      class MyClass:
        @staticmethod
        def foo(x):
          x
      """);

    var classDef = (ClassDef) root.statements().statements().get(0);
    var fooMethodDef = (FunctionDef) classDef.body().statements().get(0);
    var xExpressionStatement = (ExpressionStatement) fooMethodDef.body().statements().get(0);
    var xExpression = xExpressionStatement.expressions().get(0);

    assertThat(xExpression.typeV2()).isNotInstanceOf(SelfType.class);
    assertThat(xExpression.typeV2())
      .extracting(PythonType::unwrappedType)
      .isNotInstanceOf(SelfType.class);
  }

  @Test
  void setSelfParameterTypeWithNoParameters() {
    FileInput root = inferTypes("""
      class MyClass:
        def foo():
          pass
      """);

    var classDef = (ClassDef) root.statements().statements().get(0);
    var fooMethodDef = (FunctionDef) classDef.body().statements().get(0);
    assertThat(fooMethodDef.name().typeV2()).isInstanceOf(FunctionType.class);
  }

  @Test
  void setSelfParameterTypeWithOnlyTupleParameters() {
    // Python 2 syntax: tuple parameters are allowed
    FileInput root = inferTypes("""
      class MyClass:
        def foo((a, b)):
          pass
      """);

    var classDef = (ClassDef) root.statements().statements().get(0);
    var fooMethodDef = (FunctionDef) classDef.body().statements().get(0);
    assertThat(fooMethodDef.name().typeV2()).isInstanceOf(FunctionType.class);
  }

  @Test
  void inferParameterTypeHintedWithTypingSelf() {
    FileInput root = inferTypes("""
      from typing import Self
      class A:
        def foo(self, other: Self) -> Self:
          ...
      """);

    var classDef = (ClassDef) root.statements().statements().get(1);
    var classType = (ClassType) classDef.name().typeV2();
    var functionDef = (FunctionDef) classDef.body().statements().get(0);
    var functionType = (FunctionType) functionDef.name().typeV2();

    var otherParamType = functionType.parameters().get(1).declaredType().type();
    assertThat(otherParamType).isInstanceOf(ObjectType.class);
    var otherParamUnwrapped = otherParamType.unwrappedType();
    assertThat(otherParamUnwrapped).isInstanceOf(SelfType.class);
    assertThat(((SelfType) otherParamUnwrapped).innerType()).isEqualTo(classType);

    var returnType = functionType.returnType();
    assertThat(returnType).isInstanceOf(ObjectType.class);
    var returnTypeUnwrapped = returnType.unwrappedType();
    assertThat(returnTypeUnwrapped).isInstanceOf(SelfType.class);
    assertThat(((SelfType) returnTypeUnwrapped).innerType()).isEqualTo(classType);
  }

  @Test
  void inferParameterTypeHintedWithTypingExtensionsSelf() {
    FileInput root = inferTypes("""
      from typing_extensions import Self
      class A:
        def foo(self, other: Self) -> Self:
          ...
      """);

    var classType = TreeUtils.firstChild(root, tree -> tree instanceof ClassDef cd && "A".equals(cd.name().name()))
      .map(ClassDef.class::cast)
      .map(cd -> cd.name().typeV2())
      .get();

    var functionType = TreeUtils.firstChild(root, tree -> tree instanceof FunctionDef fd && "foo".equals(fd.name().name()))
      .map(FunctionDef.class::cast)
      .map(fd -> (FunctionType) fd.name().typeV2())
      .get();

    assertThat(functionType.parameters().get(1).declaredType().type())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOfSatisfying(SelfType.class, selfType -> assertThat(selfType.innerType()).isEqualTo(classType));

    assertThat(functionType.returnType())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOfSatisfying(SelfType.class, selfType -> assertThat(selfType.innerType()).isEqualTo(classType));
  }

  @Test
  void inferTypingExtensionsSelfAcrossModules() {
    TestProject testProject = new TestProject()
      .addModule("mod1.py", """
        from typing_extensions import Self
        class A:
          def foo(self) -> Self:
            ...
        """);
    FileInput root = testProject.inferTypes("""
        from mod1 import A
        class B(A): pass
        bar = B().foo()
        """);

    var barType = TreeUtils.firstChild(root, tree -> tree instanceof Name name && "bar".equals(name.name()))
      .map(Name.class::cast)
      .map(Name::typeV2)
      .get();

    assertThat(barType)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOfSatisfying(ClassType.class, classType -> assertThat(classType.name()).isEqualTo("B"));
  }

  @Test
  void inferTypeshedSelfReturnType() {
    var root = inferTypes("""
      import datetime
      dateFromordinal = datetime.date.fromordinal(1)
      datetimeFromordinal = datetime.datetime.fromordinal(1)

      class A(datetime.date): pass
      aFromordinal = A.fromordinal(1)

      class B(datetime.datetime): pass
      bFromordinal = B.fromordinal(1)
      """);

    Name dateFromordinal = PythonTestUtils.getFirstChild(root, t -> t.is(Tree.Kind.NAME) && "dateFromordinal".equals(((Name) t).name()));
    Name datetimeFromordinal = PythonTestUtils.getFirstChild(root, t -> t.is(Tree.Kind.NAME) && "datetimeFromordinal".equals(((Name) t).name()));
    Name aFromordinal = PythonTestUtils.getFirstChild(root, t -> t.is(Tree.Kind.NAME) && "aFromordinal".equals(((Name) t).name()));
    Name bFromordinal = PythonTestUtils.getFirstChild(root, t -> t.is(Tree.Kind.NAME) && "bFromordinal".equals(((Name) t).name()));

    PythonType dateType = PROJECT_LEVEL_TYPE_TABLE.getType("datetime.date");
    PythonType datetimeType = PROJECT_LEVEL_TYPE_TABLE.getType("datetime.datetime");
    PythonType aType = PythonTestUtils.<ClassDef>getFirstChild(root, t -> t instanceof ClassDef cd && "A".equals(cd.name().name())).name().typeV2();
    PythonType bType = PythonTestUtils.<ClassDef>getFirstChild(root, t -> t instanceof ClassDef cd && "B".equals(cd.name().name())).name().typeV2();

    assertThat(dateFromordinal.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(dateType);

    assertThat(datetimeFromordinal.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(datetimeType);

    assertThat(aFromordinal.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(aType);

    assertThat(bFromordinal.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(bType);
  }

  @Test
  void self_type_as_return_type_on_inherited_methods_with_qualified_expression() {
    FileInput fileInput = inferTypes("""
      import typing
      class A:
        def foo() -> typing.Self: ...

      class B(A):
        ...
      resultB = B().foo()
      """);

    PythonType resultB = PythonTestUtils.<Name>getFirstChild(fileInput, t -> t instanceof Name name && "resultB".equals(name.name())).typeV2();
    PythonType classTypeB = PythonTestUtils.<ClassDef>getFirstChild(fileInput, t -> t instanceof ClassDef cd && "B".equals(cd.name().name())).name().typeV2();

    assertThat(resultB)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(classTypeB);
  }

  @Test
  void self_type_as_return_type_on_inherited_methods() {
    FileInput fileInput = inferTypes("""
      from typing import Self
      class A:
        def foo() -> Self: ...

      class B(A):
        ...

      a = A()
      b = B()
      resultA = a.foo()
      resultB = b.foo()
      """);

    PythonType a = PythonTestUtils.<Name>getFirstChild(fileInput, t -> t instanceof Name name && "a".equals(name.name())).typeV2();
    PythonType b = PythonTestUtils.<Name>getFirstChild(fileInput, t -> t instanceof Name name && "b".equals(name.name())).typeV2();
    PythonType resultA = PythonTestUtils.<Name>getFirstChild(fileInput, t -> t instanceof Name name && "resultA".equals(name.name())).typeV2();
    PythonType resultB = PythonTestUtils.<Name>getFirstChild(fileInput, t -> t instanceof Name name && "resultB".equals(name.name())).typeV2();

    PythonType classTypeA = PythonTestUtils.<ClassDef>getFirstChild(fileInput, t -> t instanceof ClassDef cd && "A".equals(cd.name().name())).name().typeV2();
    PythonType classTypeB = PythonTestUtils.<ClassDef>getFirstChild(fileInput, t -> t instanceof ClassDef cd && "B".equals(cd.name().name())).name().typeV2();

    assertThat(a)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(classTypeA);

    assertThat(b)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(classTypeB);

    assertThat(resultA)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(classTypeA);

    assertThat(resultB)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(classTypeB);
  }

  @Test
  void parameterTypeHintedWithNonSelfUnresolvedImport() {
    FileInput root = inferTypes("""
      from unknown_module import SomeType
      class A:
        def foo(self, x: SomeType) -> None:
          ...
      """);

    var classDef = (ClassDef) root.statements().statements().get(1);
    var functionDef = (FunctionDef) classDef.body().statements().get(0);
    var functionType = (FunctionType) functionDef.name().typeV2();

    var xParamType = functionType.parameters().get(1).declaredType().type();
    assertThat(xParamType).isInstanceOf(ObjectType.class);
    assertThat(xParamType.unwrappedType()).isInstanceOf(UnknownType.UnresolvedImportType.class);
    assertThat(xParamType.unwrappedType()).isNotInstanceOf(SelfType.class);
  }

  @Test
  void parameterTypeHintedWithNonSelfSpecialForm() {
    FileInput root = inferTypes("""
      from typing import Final
      class A:
        def foo(self, x: Final) -> None:
          ...
      """);

    var classDef = (ClassDef) root.statements().statements().get(1);
    var functionDef = (FunctionDef) classDef.body().statements().get(0);
    var functionType = (FunctionType) functionDef.name().typeV2();

    var xParamType = functionType.parameters().get(1).declaredType().type();
    assertThat(xParamType).isInstanceOf(ObjectType.class);
    assertThat(xParamType.unwrappedType()).isNotInstanceOf(SelfType.class);
  }

  @Test
  void selfReturnTypeWithoutEnclosingClass() {
    FileInput root = inferTypes("""
      from typing import Self
      def foo() -> Self:
        ...
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(1);
    var functionType = (FunctionType) functionDef.name().typeV2();

    var returnType = functionType.returnType();
    assertThat(returnType).isNotInstanceOf(SelfType.class);
    assertThat(returnType.unwrappedType()).isNotInstanceOf(SelfType.class);
  }

  @Test
  void selfReturnTypeInNestedFunctionWithoutEnclosingClass() {
    FileInput root = inferTypes("""
      from typing import Self
      def foo():
        def bar() -> Self:
          pass
      """);

    var outerFunctionDef = (FunctionDef) root.statements().statements().get(1);
    var innerFunctionDef = (FunctionDef) outerFunctionDef.body().statements().get(0);
    var innerFunctionType = (FunctionType) innerFunctionDef.name().typeV2();

    var returnType = innerFunctionType.returnType();
    assertThat(returnType).isNotInstanceOf(SelfType.class);
    assertThat(returnType.unwrappedType()).isNotInstanceOf(SelfType.class);
  }

  @Test
  void selfReturnTypeInNestedFunctionInsideMethod() {
    FileInput root = inferTypes("""
      from typing import Self
      class A:
        def foo(self):
          def bar() -> Self:
            pass
      """);

    var classDef = (ClassDef) root.statements().statements().get(1);
    var methodDef = (FunctionDef) classDef.body().statements().get(0);
    var innerFunctionDef = (FunctionDef) methodDef.body().statements().get(0);
    var innerFunctionType = (FunctionType) innerFunctionDef.name().typeV2();

    // bar's return type should NOT be resolved to SelfType(A) because bar is not directly owned by class A
    var returnType = innerFunctionType.returnType();
    assertThat(returnType).isNotInstanceOf(SelfType.class);
    assertThat(returnType.unwrappedType()).isNotInstanceOf(SelfType.class);
  }

  @Test
  void stringLiteralTypeInference() {
    Expression root = lastExpression("""
      def bar(param: "MyParameterType") -> "MyReturnType":
        return param
      bar
      """);

    assertThat(root.typeV2()).isInstanceOf(FunctionType.class);
    FunctionType functionType = (FunctionType) root.typeV2();
    assertThat(functionType.parameters().get(0).declaredType().type().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
    assertThat(functionType.returnType().unwrappedType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void selfInUnionReturnType() {
    FileInput root = inferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self | None:
          ...
      """);

    var classDef = (ClassDef) root.statements().statements().get(1);
    var classType = (ClassType) classDef.name().typeV2();
    var functionDef = (FunctionDef) classDef.body().statements().get(0);
    var functionType = (FunctionType) functionDef.name().typeV2();

    var returnType = functionType.returnType();
    assertThat(returnType).isInstanceOf(ObjectType.class);
    var unwrappedReturnType = returnType.unwrappedType();
    assertThat(unwrappedReturnType).isInstanceOf(UnionType.class);

    var unionType = (UnionType) unwrappedReturnType;
    var candidates = unionType.candidates();
    assertThat(candidates).hasSize(2);

    var selfCandidate = candidates.stream()
      .filter(c -> c.unwrappedType() instanceof SelfType)
      .findFirst();
    assertThat(selfCandidate).isPresent();
    var selfType = (SelfType) selfCandidate.get().unwrappedType();
    assertThat(selfType.innerType()).isEqualTo(classType);

    var noneCandidate = candidates.stream()
      .filter(c -> !(c.unwrappedType() instanceof SelfType))
      .findFirst();
    assertThat(noneCandidate).isPresent();
  }

  @Test
  void selfInUnionReturnTypeWithoutEnclosingClass() {
    FileInput root = inferTypes("""
      from typing import Self
      def foo() -> Self | None:
        ...
      """);

    var functionDef = (FunctionDef) root.statements().statements().get(1);
    var functionType = (FunctionType) functionDef.name().typeV2();

    var returnType = functionType.returnType();
    assertThat(returnType).isInstanceOf(ObjectType.class);
    var unwrappedReturnType = returnType.unwrappedType();
    assertThat(unwrappedReturnType).isInstanceOf(UnionType.class);

    var unionType = (UnionType) unwrappedReturnType;
    assertThat(unionType.candidates()).noneMatch(c -> c.unwrappedType() instanceof SelfType);
  }

  @Test
  void selfOrNoneTypeAlias() {
    FileInput root = inferTypes("""
      from typing import Self
      SelfOrNone = Self | None
      class A:
        def foo(self) -> SelfOrNone:
          ...
      """);

    ClassDef classDef = getFirstDescendant(root, t -> t instanceof ClassDef cd && "A".equals(cd.name().name()));
    var classType = (ClassType) classDef.name().typeV2();
    FunctionDef functionDef = getFirstDescendant(root, t -> t instanceof FunctionDef fd && "foo".equals(fd.name().name()));
    var functionType = (FunctionType) functionDef.name().typeV2();

    assertThat(functionType.returnType())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOfSatisfying(UnionType.class, unionType ->
        assertThat(unionType.candidates())
          .hasSize(2)
          .anyMatch(c -> c instanceof SelfType st && st.innerType().equals(classType))
          .anyMatch(ObjectType.class::isInstance)
      );
  }

  @Test
  void selfOrNoneTypeAliasWithoutEnclosingClass() {
    FileInput root = inferTypes("""
      from typing import Self
      SelfOrNone = Self | None
      def foo() -> SelfOrNone:
        ...
      """);

    FunctionDef functionDef = getFirstDescendant(root, t -> t instanceof FunctionDef fd && "foo".equals(fd.name().name()));
    var functionType = (FunctionType) functionDef.name().typeV2();

    assertThat(functionType.returnType())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOfSatisfying(UnionType.class, unionType ->
        assertThat(unionType.candidates()).noneMatch(SelfType.class::isInstance)
      );
  }

  @Test
  void nestedUnionTypeAlias() {
    FileInput root = inferTypes("""
      from typing import Self
      MyType = Self | int | None
      """);

    BinaryExpression outerUnion = getFirstDescendant(root,
      t -> t instanceof BinaryExpression be && be.leftOperand() instanceof BinaryExpression);
    assertThat(outerUnion.leftOperand().typeV2()).isInstanceOf(UnionType.class);
    assertThat(outerUnion.typeV2())
      .isInstanceOfSatisfying(UnionType.class, union ->
        assertThat(union.candidates()).hasSize(3)
      );
  }

  @Test
  void bitwiseOrOnValuesDoesNotCreateUnionType() {
    FileInput root = inferTypes("""
      a = True
      b = False
      result = a | b
      """);

    AssignmentStatement assignment = getFirstDescendant(root,
      t -> t instanceof AssignmentStatement as && as.assignedValue().is(Tree.Kind.BITWISE_OR));

    assertThat(assignment.assignedValue().typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test  
  void anyResolvesToUnknown() {
    Expression anyExpression = lastExpression("""
      from typing import Any
      Any
      """);
    assertThat(anyExpression.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void typingExtensionsAnyResolvesToUnknown() {
    Expression anyExpression = lastExpression("""
      from typing_extensions import Any
      Any
      """);
    assertThat(anyExpression.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void asyncFunctionCallReturnsCoroutineType() {
    FileInput root = inferTypes("""
      async def foo() -> int:
        return 3
      foo()
      """);

    CallExpression callExpression = PythonTestUtils.getFirstChild(root, t -> t.is(Tree.Kind.CALL_EXPR));

    assertThat(callExpression.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOfSatisfying(ClassType.class, classType -> assertThat(classType.fullyQualifiedName()).isEqualTo("typing.Coroutine"));

    assertThat(((ObjectType) callExpression.typeV2()).attributes())
      .containsExactly(ObjectType.fromType(INT_TYPE));
  }

  @Test
  void awaitExpressionUnpacksCoroutineType() {
    FileInput root = inferTypes("""
      async def foo() -> int:
        return 3
      await foo()
      """);

    AwaitExpression awaitExpression = PythonTestUtils.getFirstChild(root, t -> t.is(Tree.Kind.AWAIT));

    assertThat(awaitExpression.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(INT_TYPE);
  }

  static Stream<Arguments> awaitExpressionReturnsUnknownTestCases() {
    return Stream.of(
      Arguments.of("non-coroutine return type", """
        def regular_function() -> int:
          return 3
        await regular_function()
        """),
      Arguments.of("async function without return type annotation", """
        async def foo():
          return 3
        await foo()
        """),
      Arguments.of("non-object type (class)", """
        class MyClass: ...
        await MyClass
        """)
    );
  }

  @ParameterizedTest(name = "await on {0} returns UNKNOWN")
  @MethodSource("awaitExpressionReturnsUnknownTestCases")
  void awaitExpressionReturnsUnknown(String description, String code) {
    FileInput root = inferTypes(code);

    AwaitExpression awaitExpression = PythonTestUtils.getFirstChild(root, t -> t.is(Tree.Kind.AWAIT));

    assertThat(awaitExpression.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void asyncFunctionWithoutReturnTypeAnnotation() {
    FileInput root = inferTypes("""
      async def foo():
        return 3
      foo()
      """);

    CallExpression callExpression = PythonTestUtils.getFirstChild(root, t -> t.is(Tree.Kind.CALL_EXPR));

    assertThat(callExpression.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOfSatisfying(ClassType.class, classType -> assertThat(classType.fullyQualifiedName()).isEqualTo("typing.Coroutine"));
  }

  @Test
  void conditionalExpressionTypeV2_sameTypes() {
    ConditionalExpression conditionalExpression = PythonTestUtils.getFirstChild(
      inferTypes("1 if True else 2"),
      t -> t.is(Tree.Kind.CONDITIONAL_EXPR)
    );

    assertThat(conditionalExpression.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(INT_TYPE);
  }

  @Test
  void conditionalExpressionTypeV2_differentTypes() {
    ConditionalExpression conditionalExpression = PythonTestUtils.getFirstChild(
      inferTypes("1 if True else 'hello'"),
      t -> t.is(Tree.Kind.CONDITIONAL_EXPR)
    );

    assertThat(conditionalExpression.typeV2()).isInstanceOfSatisfying(UnionType.class, unionType ->
      assertThat(unionType.candidates())
        .extracting(PythonType::unwrappedType)
        .containsExactlyInAnyOrder(INT_TYPE, STR_TYPE)
    );
  }

  @Test
  void sliceExpressionTypeV2_list() {
    SliceExpression sliceExpression = PythonTestUtils.getFirstChild(
      inferTypes("[1, 2, 3][0:2]"),
      t -> t.is(Tree.Kind.SLICE_EXPR)
    );

    assertThat(sliceExpression.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(LIST_TYPE);
  }

  @Test
  void sliceExpressionTypeV2_tuple() {
    SliceExpression sliceExpression = PythonTestUtils.getFirstChild(
      inferTypes("(1, 2, 3)[0:2]"),
      t -> t.is(Tree.Kind.SLICE_EXPR)
    );

    assertThat(sliceExpression.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(TUPLE_TYPE);
  }

  @Test
  void sliceExpressionTypeV2_string() {
    SliceExpression sliceExpression = PythonTestUtils.getFirstChild(
      inferTypes("'hello'[0:2]"),
      t -> t.is(Tree.Kind.SLICE_EXPR)
    );

    assertThat(sliceExpression.typeV2())
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(STR_TYPE);
  }

  @Test
  void sliceExpressionTypeV2_unknown() {
    SliceExpression sliceExpression = PythonTestUtils.getFirstChild(
      inferTypes("unknown_var[0:2]"),
      t -> t.is(Tree.Kind.SLICE_EXPR)
    );

    assertThat(sliceExpression.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  void argsAndKwargsParameterType() {
    FileInput root = inferTypes("""
      def foo(*args, **kwargs):
        return kwargs
      """);

    FunctionDef functionDef = PythonTestUtils.getFirstChild(root, FunctionDef.class::isInstance);
    assertThat(functionDef.parameters().all())
      .map(Parameter.class::cast)
      .satisfies(
        p -> assertThat(p.name().typeV2())
          .isInstanceOf(ObjectType.class)
          .extracting(PythonType::unwrappedType)
          .isEqualTo(TUPLE_TYPE),
        Index.atIndex(0))
      .satisfies(
        p -> assertThat(p.name().typeV2())
          .isInstanceOf(ObjectType.class)
          .extracting(PythonType::unwrappedType)
          .isEqualTo(DICT_TYPE),
        Index.atIndex(1));
  }

  @Test
  void argsAndKwargsParameterWithTypeHintsType() {
    FileInput root = inferTypes("""
      def foo(*args: int, **kwargs: str):
        return kwargs
      """);

    FunctionDef functionDef = PythonTestUtils.getFirstChild(root, FunctionDef.class::isInstance);
    assertThat(functionDef.parameters().all())
      .map(Parameter.class::cast)
      .satisfies(
        p -> assertThat(p.name().typeV2())
          .isInstanceOf(ObjectType.class)
          .satisfies(type -> assertThat(type.unwrappedType()).isEqualTo(TUPLE_TYPE))
          .isInstanceOfSatisfying(ObjectType.class, objectType -> assertThat(objectType.attributes()).isEmpty()),
        Index.atIndex(0))
      .satisfies(
        p -> assertThat(p.name().typeV2())
          .isInstanceOf(ObjectType.class)
          .satisfies(type -> assertThat(type.unwrappedType()).isEqualTo(DICT_TYPE))
          .isInstanceOfSatisfying(ObjectType.class, objectType -> assertThat(objectType.attributes()).isEmpty()),
        Index.atIndex(1));
  }

  @Test
  void testModuleLeakingIntoProjectLevelTypeTable() {
    var project = new TestProject();
    project.addModule("package/__init__.py", "");

    var finalModuleCode = """
      class SuperClass: ...
      class AClass(SuperClass):
        ...
      """;
    project.addModule("package/finalModule.py", finalModuleCode);
    PythonFile finalModulePythonFile = pythonFile("finalModule.py");
    new PythonVisitorContext.Builder(project.inferTypes(finalModuleCode), finalModulePythonFile)
      .typeTable(project.projectLevelTypeTable())
      .projectLevelSymbolTable(project.projectLevelSymbolTable())
      .packageName("package")
      .build();

    PythonType bType = project.projectLevelTypeTable().getType("package.finalModule.AClass");
    assertThat(bType).isInstanceOfSatisfying(ClassType.class, classType -> {
      assertThat(classType.superClasses())
        .satisfies(typeWrapper -> assertThat(typeWrapper)
          .isInstanceOf(LazyTypeWrapper.class), Index.atIndex(0));
    });
  }
}
