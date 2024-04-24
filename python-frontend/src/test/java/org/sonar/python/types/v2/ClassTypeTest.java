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
package org.sonar.python.types.v2;

import java.util.List;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;
import org.sonar.python.semantic.v2.UsageV2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;

public class ClassTypeTest {

  @Test
  void no_parents() {
    ClassType classType = classType("class C: ...");
    assertThat(classType.superClasses()).isEmpty();
    assertThat(classType.hasUnresolvedHierarchy()).isFalse();
    assertThat(classType.hasMetaClass()).isFalse();
    assertThat(classType.hasUnresolvedHierarchy()).isFalse();
    // TODO: not correct
    assertThat(classType.key()).isEqualTo("C[]");

    assertThat(classType.hasMember("__call__")).isEqualTo(TriBool.TRUE);
    assertThat(classType.hasMember("unknown")).isEqualTo(TriBool.UNKNOWN);
    assertThat(classType.instancesHaveMember("__call__")).isEqualTo(TriBool.FALSE);
    assertThat(classType.instancesHaveMember("unknown")).isEqualTo(TriBool.FALSE);
  }

  @Test
  void equals_test() {
    ClassType classType1 = classType("class C: ...");
    ClassType classType2 = classType("class C: ...");
    ClassType classType3 = classType("class C(B): ...");
    ClassType classType4 = classType("class C(metaclass=MyMeta): ...");
    ClassType classType5 = classType("class D: ...");
    ClassType classType6 = classType("""
          class C:
            def foo(): ...
          """
    );

    assertThat(classType1).isEqualTo(classType1);
    assertThat(classType1).isEqualTo(classType2);
    assertThat(classType1).isNotEqualTo(classType3);
    assertThat(classType1).isNotEqualTo(classType4);
    assertThat(classType1).isNotEqualTo(classType5);
    assertThat(classType1).isNotEqualTo(classType6);
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
    assertThat(classB.superClasses()).containsExactlyInAnyOrder(classC);

    //assertThat(fileInput.globalVariables()).hasSize(2);
    //assertThat(fileInput.globalVariables()).containsExactlyInAnyOrder(symbol, parentSymbol);
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
    assertThat(classB.superClasses()).containsExactlyInAnyOrder(classC, classA);

/*    assertThat(fileInput.globalVariables()).hasSize(3);
    assertThat(fileInput.globalVariables()).containsExactlyInAnyOrder(symbol, parentSymbol, parentSymbol2);*/
  }

  @Test
  void unknown_parent() {
    List<ClassType> classTypes = classTypes(
      "class B(C): ..."
    );
    ClassType classC = classTypes.get(0);
    assertThat(classC.superClasses()).containsExactly(PythonType.UNKNOWN);
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
    // FIXME: ensure builtin parent is resolved
    assertThat(classB.hasUnresolvedHierarchy()).isTrue();
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
    fileInput.accept(new SymbolTableBuilderV2());
    fileInput.accept(new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())));

    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(0);
    PythonType pythonType = classDef.name().typeV2();
    assertThat(pythonType).isInstanceOf(ClassType.class);

    SymbolV2 symbol = classDef.name().symbolV2();
    assertThat(symbol.name()).isEqualTo("C");
    assertThat(symbol.usages()).hasSize(2);
    assertThat(symbol.usages()).extracting(UsageV2::isBindingUsage).containsExactly(true, true);
  }

  @Test
  void multiple_bindings_2() {
    FileInput fileInput = parse(
      "C = \"hello\"",
      "class C: ",
      "  pass"
    );
    fileInput.accept(new SymbolTableBuilderV2());
    fileInput.accept(new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())));

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
    // TODO: ObjectType[PythonType.UNKNOWN] vs PythonType.UNKNOWN
    //assertThat(classType.superClasses()).containsExactly(PythonType.UNKNOWN);
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
    // TODO: ObjectType[PythonType.UNKNOWN] vs PythonType.UNKNOWN
    //assertThat(classType.superClasses()).containsExactly(PythonType.UNKNOWN);
    assertThat(classType.hasUnresolvedHierarchy()).isTrue();
  }

  @Test
  void unpacking_expression_as_parent() {
    List<ClassType> classTypes = classTypes(
      "foo = (Something, SomethingElse)",
      "class C(*foo): ",
      "  pass");
    ClassType classType = classTypes.get(0);
    assertThat(classType.superClasses()).containsExactly(PythonType.UNKNOWN);
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

    // FIXME: fix the test
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
    assertThat(classB.superClasses()).extracting(PythonType::name).containsExactly("A");
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
    // TODO: Have assertions on symbol to check binding usages
  }

  @Test
  void class_with_nonlocal_statement() {
    ClassType classType = classType(
      "nonlocal B",
      "class B(): ",
      "  pass");
    assertThat(classType.name()).isEqualTo("B");
    // TODO: Have assertions on symbol to check binding usages
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
    ClassType classType = classTypes(
      "class A:",
      "  def meth(): pass",
      "class B(A): ",
      "  def foo(): pass").get(1);

    assertThat(classType.members()).hasSize(1);
    ClassType classA = (ClassType) classType.superClasses().get(0);
    assertThat(classA.members()).hasSize(1);
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
    PythonType foo = classType.resolveMember("foo");
    assertThat(foo).isNotNull();
    assertThat(classType.resolveMember("bar")).isNull();
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
    PythonType foo = classType.resolveMember("foo");
    assertThat(foo).isNotNull();
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
  @Disabled("Add assertions, API for decorators")
  void has_decorators() {
    ClassType classA = classType(
      "@foo",
      "class A: ..."
    );
//    assertThat(classA.hasDecorators()).isTrue();

    classA = classType(
      "class A: ..."
    );
//    assertThat(classA.hasDecorators()).isFalse();
  }

  @Test
  void type_annotations_scope() {
    FileInput fileInput = PythonTestUtils.parse(
      "class Foo:",
      "    class Inner: ...",
      "    def f(self, x: Inner) -> Inner:",
      "        x: Inner"
    );
    fileInput.accept(new SymbolTableBuilderV2());
    fileInput.accept(new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())));
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
    ClassType classType = classType("class A: ...");
    ClassTypeBuilder classTypeBuilder = new ClassTypeBuilder().setName("A");
    assertThat(classTypeBuilder.build()).isEqualTo(classType);
  }

  public static ClassType classType(String... code) {
    return classTypes(code).get(0);
  }

  public static List<ClassType> classTypes(String... code) {
    FileInput fileInput = parseWithoutSymbols(code);
    fileInput.accept(new SymbolTableBuilderV2());
    fileInput.accept(new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())));
    return PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF))
      .stream()
      .map(ClassDef.class::cast)
      .map(ClassDef::name)
      .map(Name::typeV2)
      .map(ClassType.class::cast)
      .toList();
  }
}
