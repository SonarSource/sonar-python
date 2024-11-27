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
import org.assertj.core.groups.Tuple;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.LocationInFile;
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
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.ExpressionStatementImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.tree.TupleImpl;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.SimpleTypeWrapper;
import org.sonar.python.types.v2.TypeOrigin;
import org.sonar.python.types.v2.TypeSource;
import org.sonar.python.types.v2.TypeWrapper;
import org.sonar.python.types.v2.UnionType;
import org.sonar.python.types.v2.UnknownType;
import org.sonar.python.types.v2.UnknownType.UnresolvedImportType;

import static org.assertj.core.api.Assertions.assertThat;
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
    FileInput fileInput = inferTypes("""
      random[2]()
      """);
    CallExpression callExpression = ((CallExpression) ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0));

    CallExpression callExpressionSpy = Mockito.spy(callExpression);
    Expression calleeSpy = Mockito.spy(callExpression.callee());
    FunctionType functionType = new FunctionType("foo", "my_package.foo", List.of(), List.of(), List.of(), new SimpleTypeWrapper(new ObjectType(INT_TYPE)), TypeOrigin.STUB,
      false, false, false, false, null, null);
    Mockito.when(calleeSpy.typeV2()).thenReturn(functionType);
    Mockito.when(callExpressionSpy.callee()).thenReturn(calleeSpy);

    var resultType = callExpressionSpy.typeV2();
    assertThat(resultType.typeSource()).isEqualTo(TypeSource.EXACT);
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

    PythonType nonePythonType = new ObjectType(NONE_TYPE, List.of(), List.of());
    PythonType myClassPythonType = ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();

    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    for (CallExpression call : calls) {
      PythonType calleeType = call.callee().typeV2();
      assertThat(calleeType).isInstanceOf(UnionType.class);
      assertThat(calleeType.isCompatibleWith(nonePythonType)).isTrue();
      assertThat(calleeType.isCompatibleWith(myClassPythonType)).isTrue();
    }
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
    assertThat(acosType1.candidates()).allMatch(p -> p instanceof FunctionType);
    assertThat(acosType1.candidates())
      .map(FunctionType.class::cast)
      .extracting(FunctionType::returnType)
      .extracting(PythonType::unwrappedType)
      .containsExactly(PythonType.UNKNOWN, PythonType.UNKNOWN, PythonType.UNKNOWN, PythonType.UNKNOWN);

    UnionType acosType2 = (UnionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(acosType2.candidates()).allMatch(p -> p instanceof FunctionType);
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
    // Declared return types of local functions are currently not stored in the project level symbol table
    assertThat(xType).isEqualTo(PythonType.UNKNOWN);
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
    UnionType unionType = new UnionType(Set.of(PythonType.UNKNOWN));
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
    UnionType unionType = new UnionType(Set.of());
    Name mock = Mockito.mock(Name.class);
    Mockito.when(mock.typeV2()).thenReturn(unionType);
    Mockito.doReturn(mock).when(callExpressionSpy).callee();

    assertThat(callExpressionSpy.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void return_type_of_call_expression_inconsistent_3() {
    FileInput fileInput = inferTypes(
      """
        foo()
        """
    );
    CallExpression callExpression = ((CallExpression) ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0));
    CallExpression callExpressionSpy = Mockito.spy(callExpression);

    // Inconsistent union type, should not happen
    UnionType unionType = new UnionType(Set.of(INT_TYPE));
    Name mock = Mockito.mock(Name.class);
    Mockito.when(mock.typeV2()).thenReturn(unionType);
    Mockito.doReturn(mock).when(callExpressionSpy).callee();

    assertThat(callExpressionSpy.typeV2().unwrappedType()).isEqualTo(INT_TYPE);
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
    ModuleType builtinModule = symbolsModuleTypeProvider.createBuiltinModule();
    symbolsModuleTypeProvider.convertModuleType(List.of("typing"), builtinModule);

    ClassSymbol symbol = Mockito.mock(ClassSymbolImpl.class);
    Mockito.when(symbol.kind()).thenReturn(Symbol.Kind.OTHER);
    assertThat(PROJECT_LEVEL_TYPE_TABLE.lazyTypesContext().getOrCreateLazyType("typing.Iterable.unknown").resolve()).isEqualTo(PythonType.UNKNOWN);
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
    assertThat(fType).isSameAs(PythonType.UNKNOWN);
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

    // foo should be UNKNOWN, but is resolved to foo because wildcard imports don't respect scoping. See SONARPY-2357
    assertThat(fooExpr.typeV2()).isNotEqualTo(PythonType.UNKNOWN);
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

  private static Map<SymbolV2, Set<PythonType>> inferTypesBySymbol(String lines) {
    FileInput root = parse(lines);
    var symbolTable = new SymbolTableBuilderV2(root).build();
    var typeInferenceV2 = new TypeInferenceV2(PROJECT_LEVEL_TYPE_TABLE, pythonFile, symbolTable, "");
    return typeInferenceV2.inferTypes(root);
  }

  private static FileInput inferTypes(String lines) {
    return inferTypes(lines, PROJECT_LEVEL_TYPE_TABLE);
  }

  private static FileInput inferTypes(String lines, ProjectLevelTypeTable projectLevelTypeTable) {
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

  private static class TestProject {
    private final ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();

    public TestProject addModule(String moduleName, String code) {
      FileInput tree = parseWithoutSymbols(code);
      projectLevelSymbolTable.addModule(tree, "", pythonFile(moduleName));
      return this;
    }

    public Expression lastExpression(String code) {
      ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
      return TypeInferenceV2Test.lastExpression(code, projectLevelTypeTable);
    }

    public TupleImpl lastExpressionAsTuple(String code) {
      Expression lastExpr = lastExpression(code);
      assertThat(lastExpr).isInstanceOf(TupleImpl.class);
      return (TupleImpl) lastExpr;
    }
  }
}
