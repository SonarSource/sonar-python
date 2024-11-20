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
package org.sonar.python.types.v2;

import java.util.List;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;
import org.sonar.python.semantic.v2.UsageV2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;

public class ClassTypeTest {

  static PythonFile pythonFile = PythonTestUtils.pythonFile("");

  @Test
  void no_parents() {
    ClassType classType = classType("class C: ...");
    assertThat(classType.superClasses()).isEmpty();
    assertThat(classType.hasUnresolvedHierarchy()).isFalse();
    assertThat(classType.hasMetaClass()).isFalse();
    assertThat(classType.hasUnresolvedHierarchy()).isFalse();
    // TODO: SONARPY-1874 correctly represent keys
    assertThat(classType.key()).isEqualTo("C[]");
    assertThat(classType).hasToString("ClassType[C]");

    assertThat(classType.hasMember("__call__")).isEqualTo(TriBool.TRUE);
    assertThat(classType.hasMember("unknown")).isEqualTo(TriBool.UNKNOWN);
    assertThat(classType.instancesHaveMember("__call__")).isEqualTo(TriBool.FALSE);
    assertThat(classType.instancesHaveMember("unknown")).isEqualTo(TriBool.FALSE);

    assertThat(classType.displayName()).contains("type");
    assertThat(classType.instanceDisplayName()).contains("C");
    assertThat(classType.unwrappedType()).isEqualTo(classType);

    String fileId = SymbolUtils.pathOf(pythonFile).toString();
    assertThat(classType.definitionLocation()).contains(new LocationInFile(fileId, 1, 6, 1, 7));
  }

  @Test
  void simple_member() {
    ClassType classType = classType("""
          class C:
            def foo(): ...
          """
    );

    assertThat(classType.instancesHaveMember("foo")).isEqualTo(TriBool.TRUE);
    assertThat(classType.instancesHaveMember("bar")).isEqualTo(TriBool.FALSE);
  }

  @Test
  void local_parent() {
    List<ClassType> classTypes = classTypes(
      "class C: ",
      "  pass",
      "class B(C): ",
      "  pass");
    ClassType classC = classTypes.get(0);
    ClassType classB = classTypes.get(1);
    assertThat(classB.superClasses()).hasSize(1);
    assertThat(classB.superClasses()).extracting(TypeWrapper::type).containsExactlyInAnyOrder(classC);
  }

  @Test
  void multiple_local_parents() {
    List<ClassType> classTypes = classTypes(
      "class C: ",
      "  pass",
      "class A:",
      "  pass",
      "class B(C, A): ",
      "  pass");
    ClassType classC = classTypes.get(0);
    ClassType classA = classTypes.get(1);
    ClassType classB = classTypes.get(2);
    assertThat(classB.superClasses()).hasSize(2);
    assertThat(classB.superClasses()).extracting(TypeWrapper::type).containsExactlyInAnyOrder(classC, classA);
  }

  @Test
  void unknown_parent() {
    List<ClassType> classTypes = classTypes(
      "class B(C): ..."
    );
    ClassType classC = classTypes.get(0);
    assertThat(classC.superClasses()).extracting(TypeWrapper::type).containsExactly(PythonType.UNKNOWN);
    assertThat(classC.hasUnresolvedHierarchy()).isTrue();
    assertThat(classC.hasMember("unknown")).isEqualTo(TriBool.UNKNOWN);
    assertThat(classC.instancesHaveMember("unknown")).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void named_tuple_is_type() {
    List<ClassType> classTypes = classTypes(
      "class NamedTuple: ..."
    );
    ClassType classC = classTypes.get(0);
    // TODO: SONARPY-1666 should properly fix this
    assertThat(classC.instancesHaveMember("unknown")).isEqualTo(TriBool.TRUE);
  }

  @Test
  void builtin_parent() {
    List<ClassType> classTypes = classTypes(
      "class C: ...",
      "class B(C, BaseException): ..."
    );
    ClassType classB = classTypes.get(1);
    assertThat(classB.superClasses()).hasSize(2);
    assertThat(classB.hasUnresolvedHierarchy()).isFalse();
    var baseExceptionType = classB.superClasses().get(1).type();
    assertThat(baseExceptionType)
      .isInstanceOf(ClassType.class)
      .extracting(PythonType::name)
      .isEqualTo("BaseException");

    var baseExceptionClassType = (ClassType) baseExceptionType;
    assertThat(baseExceptionClassType.members()).hasSize(10);
  }

  @Test
  void builtin_parent_with_unknown() {
    List<ClassType> classTypes = classTypes(
      "class C: ",
      "  pass",
      "class B(C, BaseException, unknown): ",
      "  pass");
    ClassType classB = classTypes.get(1);
    assertThat(classB.superClasses()).hasSize(3);
    assertThat(classB.hasUnresolvedHierarchy()).isTrue();
  }

  @Test
  void multiple_bindings() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "C = \"hello\"");
    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();
    new TypeInferenceV2(PROJECT_LEVEL_TYPE_TABLE, pythonFile, symbolTable, "").inferTypes(fileInput);

    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(0);
    PythonType pythonType = classDef.name().typeV2();
    assertThat(pythonType).isInstanceOf(ClassType.class);

    SymbolV2 symbol = classDef.name().symbolV2();
    assertThat(symbol.name()).isEqualTo("C");
    assertThat(symbol.usages()).hasSize(2);
    assertThat(symbol.usages()).extracting(UsageV2::isBindingUsage).containsExactly(true, true);

    var moduleSymbols = symbolTable.getSymbolsByRootTree(fileInput);
    assertThat(moduleSymbols).hasSize(1).contains(symbol);
  }

  @Test
  void multiple_bindings_2() {
    FileInput fileInput = parse(
      "C = \"hello\"",
      "class C: ",
      "  pass"
    );
    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();
    new TypeInferenceV2(PROJECT_LEVEL_TYPE_TABLE, pythonFile, symbolTable, "").inferTypes(fileInput);

    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    PythonType pythonType = classDef.name().typeV2();
    assertThat(pythonType).isInstanceOf(ClassType.class);

    SymbolV2 symbol = classDef.name().symbolV2();
    assertThat(symbol.name()).isEqualTo("C");
    assertThat(symbol.usages()).hasSize(2);
    assertThat(symbol.usages()).extracting(UsageV2::isBindingUsage).containsExactly(true, true);
  }

  @Test
  void call_expression_argument() {
    List<ClassType> classTypes = classTypes(
      "def foo():",
      "  pass",
      "class C(foo()): ",
      "  pass");
    ClassType classType = classTypes.get(0);
    assertThat(classType.superClasses()).extracting(TypeWrapper::type).containsExactly(PythonType.UNKNOWN);
    assertThat(classType.hasUnresolvedHierarchy()).isTrue();
  }

  @Test
  void parent_is_not_a_class() {
    List<ClassType> classTypes = classTypes(
      "def foo():",
      "  pass",
      "A = foo()",
      "class C(A): ",
      "  pass");
    ClassType classType = classTypes.get(0);
    assertThat(classType.superClasses()).hasSize(1);
    assertThat(classType.superClasses()).extracting(TypeWrapper::type).containsExactly(PythonType.UNKNOWN);
    assertThat(classType.hasUnresolvedHierarchy()).isTrue();
  }

  @Test
  void unpacking_expression_as_parent() {
    List<ClassType> classTypes = classTypes(
      "foo = (Something, SomethingElse)",
      "class C(*foo): ",
      "  pass");
    ClassType classType = classTypes.get(0);
    assertThat(classType.superClasses()).extracting(TypeWrapper::type).containsExactly(PythonType.UNKNOWN);
    assertThat(classType.hasUnresolvedHierarchy()).isTrue();
  }

  @Test
  void parent_has_multiple_bindings() {
    List<ClassType> classTypes = classTypes(
      "class C: ",
      "  pass",
      "C = \"hello\"",
      "class B(C): ",
      "  pass");
    ClassType classType = classTypes.get(1);
    assertThat(classType.hasUnresolvedHierarchy()).isTrue();
  }

  @Test
  void simple_metaclass() {
    List<ClassType> classTypes = classTypes(
      "class A: ",
      "  pass",
      "class B(metaclass=A): ",
      "  pass");

    ClassType classA = classTypes.get(0);
    ClassType classB = classTypes.get(1);

    assertThat(classB.hasUnresolvedHierarchy()).isFalse();
    assertThat(classB.superClasses()).isEmpty();
    assertThat(classB.hasMetaClass()).isTrue();
    assertThat(classB.metaClasses()).containsExactly(classA);
    assertThat(classB.instancesHaveMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  @Disabled("TODO: support metaclasse names/types")
  void other_metaclasses() {
    ClassType classC = classTypes(
      "class A: ",
      "  pass",
      "class B(metaclass=A): ",
      "  pass",
      "class C(B):",
      "  pass").get(2);

    assertThat(classC.instancesHaveMember("foo")).isEqualTo(TriBool.UNKNOWN);

    classC = classTypes(
      "from abc import ABCMeta",
      "class B(metaclass=ABCMeta): ",
      "  pass",
      "class C(B):",
      "  pass").get(1);

    assertThat(classC.instancesHaveMember("foo")).isEqualTo(TriBool.UNKNOWN);

    classC = classTypes(
      "from abc import ABCMeta",
      "class Factory: ...",
      "class A(metaclass=Factory): ",
      "  pass",
      "class B(A, metaclass=ABCMeta): ",
      "  pass",
      "class C(B):",
      "  pass").get(3);

    assertThat(classC.instancesHaveMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void unknown_argument() {
    ClassType classType = classType("class C(not_meta_class=X): ...");
    assertThat(classType.superClasses()).isEmpty();
    assertThat(classType.metaClasses()).isEmpty();
    // TODO: Maybe we want a different behavior here
    assertThat(classType.hasUnresolvedHierarchy()).isFalse();
  }

  @Test
  @Disabled("TODO: handle Python 2 style metaclasses")
  void defines_metaclass_python_2() {
    List<ClassType> classTypes = classTypes(
      "class A: ",
      "  pass",
      "class B(): ",
      "  __metaclass__ = A");
    ClassType classB = classTypes.get(1);
    assertThat(classB.hasUnresolvedHierarchy()).isFalse();
    assertThat(classB.superClasses()).isEmpty();
    assertThat(classB.hasMetaClass()).isTrue();

    assertThat(classB.instancesHaveMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  @Disabled("TODO: support attrs")
  void defines_attrs() {
    List<ClassType> classTypes = classTypes(
      "class A: ",
      "  pass",
      "class B(A, attrs=...): ",
      "  pass");
    ClassType classB = classTypes.get(1);
    assertThat(classB.hasUnresolvedHierarchy()).isFalse();
    assertThat(classB.superClasses()).hasSize(1);
    assertThat(classB.superClasses()).extracting(TypeWrapper::type).extracting(PythonType::name).containsExactly("A");
    assertThat(classB.hasMetaClass()).isFalse();
    assertThat(classB.instancesHaveMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void class_with_global_statement() {
    ClassType classType = classType(
      "global B",
      "class B(): ",
      "  pass");
    assertThat(classType.name()).isEqualTo("B");
  }

  @Test
  void class_with_nonlocal_statement() {
    ClassType classType = classType(
      "nonlocal B",
      "class B(): ",
      "  pass");
    assertThat(classType.name()).isEqualTo("B");
  }

  @Test
  void class_members_empty() {
    ClassType classType = classType(
      "class C: ",
      "  pass");
    assertThat(classType.members()).isEmpty();
  }

  @Test
  @Disabled("TODO: meanginful assertions on members")
  void class_members() {
    ClassType classType = classType(
      "class C: ",
      "  def foo(): pass");
    assertThat(classType.members()).hasSize(1);

    classType = classType(
      "class C: ",
      "  bar = 42");
    assertThat(classType.members()).hasSize(1);

  }

  @Test
  @Disabled("TODO: meanginful assertions on members")
  void duplicated_class_member_self() {
    ClassType classType = classType(
      "class C:",
      "  def f(self): ...",
      "  def g(self): ",
      "    self.f()"
    );
    assertThat(classType.members()).hasSize(1);

    classType = classType(
      "class A:",
      "  def __init__(self):",
      "    self.foo = []",
      "  def foo(self): ..."
    );
    assertThat(classType.members()).hasSize(1);
  }

  @Test
  void class_members_with_inheritance() {
    ClassType classB = classTypes(
      "class A:",
      "  def meth(): pass",
      "class B(A): ",
      "  def foo(): pass").get(1);

    assertThat(classB.members()).hasSize(1);
    ClassType classA = (ClassType) classB.superClasses().get(0).type();
    assertThat(classA.members()).hasSize(1);

    assertThat(classB.resolveMember("foo")).isPresent();
    assertThat(classB.resolveMember("meth")).isPresent();
    assertThat(classB.resolveMember("unkown")).isNotPresent();
  }

  @Test
  void class_members_multiple_inheritance() {
    // See https://docs.python.org/3/howto/mro.html#python-2-3-mro
    List<ClassType> classTypes = classTypes(
      """
        class A:
          def foo(param): ...
        class B:
          def foo(): ...
        class C(A, B): ...
        """
    );
    ClassType classA = classTypes.get(0);
    ClassType classB = classTypes.get(1);
    ClassType classC = classTypes.get(2);

    PythonType fooA = classA.resolveMember("foo").get();
    PythonType fooB = classB.resolveMember("foo").get();
    PythonType fooC = classC.resolveMember("foo").get();

    assertThat(fooC).isSameAs(fooA).isNotSameAs(fooB);
  }

  @Test
  void classTypeAmbiguousMember() {
    ClassType classType = classType("""
      class MyClass:
        def foo(param): ...
        def foo(param, other_param): ...
      """);
    assertThat(classType.resolveMember("foo")).contains(PythonType.UNKNOWN);
    assertThat(classType.resolveMember("bar")).isEmpty();
  }

  @Test
  @Disabled("TODO: resolve static members")
  void static_member_usages() {
    ClassType classType = classType(
      "class A:",
      "  foo = 42",
      "  def __init__(self): ",
      "    A.foo",
      "    A.foo = 0",
      "    A.bar"
    );
    assertThat((classType.resolveMember("foo"))).isPresent();
    assertThat(classType.resolveMember("bar")).isNotPresent();
  }

  @Test
  @Disabled("TODO: resolve static members")
  void inherited_static_member() {
    ClassType classType = classType(
      "class A:",
      "  foo = 42",
      "class B(A): pass",
      "B.foo"
    );

    assertThat(classType.instancesHaveMember("foo")).isEqualTo(TriBool.TRUE);
    assertThat(classType.resolveMember("foo")).isPresent();
  }

  @Test
  @Disabled("TODO: resolve types for multiple definitions (UnionType)")
  void inherits_from_ambiguous_symbol() {
    ClassType classB = classTypes(
      "if x:",
      "  class A: ...",
      "else:",
      "  class A:",
      "    def foo(): ...",
      "class B(A): ..."
    ).get(2);

    assertThat(classB.resolveMember("foo")).isNull();
    assertThat(classB.hasUnresolvedHierarchy()).isTrue();
    assertThat(classB.instancesHaveMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  @Disabled("TODO: fix inheritance")
  void inherits_from_function_call() {
    ClassType classB = classTypes(
      "class A:",
      "  def foo(): ...",
      "def func(): return A",
      "class B(func()): ..."
    ).get(1);

    assertThat(classB.resolveMember("foo")).isNull();
    assertThat(classB.instancesHaveMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void has_decorators() {
    ClassType classA = classType(
      "@foo",
      "class A: ..."
    );
    assertThat(classA.hasDecorators()).isTrue();
    assertThat(classA.instancesHaveMember("bar")).isEqualTo(TriBool.UNKNOWN);

    classA = classType(
      "@foo()",
      "class A: ..."
    );
    assertThat(classA.hasDecorators()).isTrue();
    assertThat(classA.instancesHaveMember("bar")).isEqualTo(TriBool.UNKNOWN);

    classA = classType(
      "@foo.bar()[qix()]",
      "class A: ..."
    );
    assertThat(classA.hasDecorators()).isTrue();
    assertThat(classA.instancesHaveMember("bar")).isEqualTo(TriBool.UNKNOWN);

    classA = classType(
      "class A: ..."
    );
    assertThat(classA.hasDecorators()).isFalse();
    assertThat(classA.instancesHaveMember("bar")).isEqualTo(TriBool.FALSE);
  }

  @Test
  void type_annotations_scope() {
    FileInput fileInput = PythonTestUtils.parse(
      "class Foo:",
      "    class Inner: ...",
      "    def f(self, x: Inner) -> Inner:",
      "        x: Inner"
    );
    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();
    new TypeInferenceV2(PROJECT_LEVEL_TYPE_TABLE, pythonFile, symbolTable, "").inferTypes(fileInput);
    ClassDef firstDef = (ClassDef) fileInput.statements().statements().get(0);
    ClassDef innerClass = (ClassDef) firstDef.body().statements().get(0);
    FunctionDef functionDef = (FunctionDef) firstDef.body().statements().get(1);

    SymbolV2 innerClassSymbolV2 = innerClass.name().symbolV2();

    // The scope of the function parameters and return type annotations is the parent scope of the function
    Name returnTypeAnnotationName = (Name) functionDef.returnTypeAnnotation().expression();
    assertThat(returnTypeAnnotationName.symbolV2()).isEqualTo(innerClassSymbolV2);

    Name paramAnnotationName = (Name) functionDef.parameters().nonTuple().get(1).typeAnnotation().expression();
    assertThat(paramAnnotationName.symbolV2()).isEqualTo(innerClassSymbolV2);

    // In the function body the scope is the scope of the function, where the inner class needs to be referenced through `self`
    Name varAnnotationName = ((Name) ((AnnotatedAssignment) functionDef.body().statements().get(0)).annotation().expression());
    assertThat(varAnnotationName.symbolV2()).isNull();
  }


  @Test
  void builder() {
    ClassTypeBuilder classTypeBuilder = new ClassTypeBuilder("A", "mod.A");
    assertThat(classTypeBuilder.build()).extracting(ClassType::name).isEqualTo("A");
  }

  @Test
  void displayName() {
    ClassType classType = new ClassType("MyClass", "mymod.MyClass");
    assertThat(classType.instanceDisplayName()).contains("MyClass");
    assertThat(classType.displayName()).contains("type");
  }

  public static ClassType classType(String... code) {
    return classTypes(code).get(0);
  }

  public static List<ClassType> classTypes(String... code) {
    FileInput fileInput = parseWithoutSymbols(code);
    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();
    new TypeInferenceV2(PROJECT_LEVEL_TYPE_TABLE, pythonFile, symbolTable, "").inferTypes(fileInput);
    return PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF))
      .stream()
      .map(ClassDef.class::cast)
      .map(ClassDef::name)
      .map(Name::typeV2)
      .map(ClassType.class::cast)
      .toList();
  }
}
