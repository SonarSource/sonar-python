/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.types;

import java.util.Collections;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.tree.TreeUtils;

import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.InferredTypes.or;
import static org.sonar.python.types.InferredTypes.runtimeBuiltinType;

class RuntimeTypeTest {

  private final ClassSymbolImpl a = new ClassSymbolImpl("a", "a");
  private final ClassSymbolImpl b = new ClassSymbolImpl("b", "b");
  private final ClassSymbolImpl c = new ClassSymbolImpl("c", "c");

  @Test
  void isIdentityComparableWith() {
    RuntimeType aType = new RuntimeType(a);
    RuntimeType bType = new RuntimeType(b);
    RuntimeType cType = new RuntimeType(c);

    assertThat(aType.isIdentityComparableWith(bType)).isFalse();
    assertThat(aType.isIdentityComparableWith(aType)).isTrue();
    assertThat(aType.isIdentityComparableWith(new RuntimeType(a))).isTrue();

    assertThat(aType.isIdentityComparableWith(AnyType.ANY)).isTrue();

    assertThat(aType.isIdentityComparableWith(or(aType, bType))).isTrue();
    assertThat(aType.isIdentityComparableWith(or(cType, bType))).isFalse();

    assertThat(aType.isIdentityComparableWith(new DeclaredType(a))).isTrue();
    assertThat(aType.isIdentityComparableWith(new DeclaredType(b))).isTrue();
  }

  @Test
  void member() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    SymbolImpl foo = new SymbolImpl("foo", null);
    x.addMembers(singletonList(foo));
    assertThat(new RuntimeType(x).canHaveMember("foo")).isTrue();
    assertThat(new RuntimeType(x).canHaveMember("bar")).isFalse();
    assertThat(new RuntimeType(x).declaresMember("foo")).isTrue();
    assertThat(new RuntimeType(x).declaresMember("bar")).isFalse();
    assertThat(new RuntimeType(x).resolveMember("foo")).contains(foo);
    assertThat(new RuntimeType(x).resolveMember("bar")).isEmpty();

    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    x1.addSuperClass(x);
    x1.addMembers(singletonList(new SymbolImpl("bar", null)));
    assertThat(new RuntimeType(x1).canHaveMember("foo")).isTrue();
    assertThat(new RuntimeType(x1).canHaveMember("bar")).isTrue();
    assertThat(new RuntimeType(x1).resolveMember("foo")).contains(foo);

    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    assertThat(new RuntimeType(y).canHaveMember("foo")).isFalse();
  }

  @Test
  void overridden_symbol() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    SymbolImpl fooX = new SymbolImpl("foo", null);
    x.addMembers(singletonList(fooX));
    assertThat(new RuntimeType(x).resolveMember("foo")).contains(fooX);

    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    SymbolImpl fooX1 = new SymbolImpl("foo", null);
    x1.addSuperClass(x);
    x1.addMembers(singletonList(fooX1));
    assertThat(new RuntimeType(x1).resolveMember("foo")).contains(fooX1);

    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    y.addMembers(singletonList(new SymbolImpl("foo", null)));
    ClassSymbolImpl z = new ClassSymbolImpl("z", "z");
    z.addSuperClass(x);
    z.addSuperClass(y);
    // TODO should be empty when multiple superclasses have the same member name
    assertThat(new RuntimeType(z).resolveMember("foo")).contains(fooX);
  }

  @Test
  void cycle_between_super_types() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    ClassSymbolImpl z = new ClassSymbolImpl("z", "z");
    x.addSuperClass(y);
    y.addSuperClass(z);
    z.addSuperClass(x);
    assertThat(new RuntimeType(x).canHaveMember("foo")).isFalse();
  }

  @Test
  void unresolved_type_hierarchy() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    x.setHasSuperClassWithoutSymbol();
    assertThat(new RuntimeType(x).canHaveMember("foo")).isTrue();

    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    y.addSuperClass(x);
    assertThat(new RuntimeType(y).canHaveMember("foo")).isTrue();
  }

  @Test
  void test_equals() {
    RuntimeType aType = new RuntimeType(a);
    assertThat(aType.equals(aType)).isTrue();
    assertThat(aType.equals(new RuntimeType(a))).isTrue();
    assertThat(aType.equals(new RuntimeType(b))).isFalse();
    assertThat(aType.equals(a)).isFalse();
    assertThat(aType.equals(null)).isFalse();

    ClassSymbolImpl aWithSuperClass = new ClassSymbolImpl("a", "a");
    aWithSuperClass.addSuperClass(b);
    RuntimeType aTypeWithSuperClass = new RuntimeType(aWithSuperClass);
    ClassSymbolImpl aWithSuperClass2 = new ClassSymbolImpl("a", "a");
    aWithSuperClass2.addSuperClass(b);
    RuntimeType aTypeWithSuperClass2 = new RuntimeType(aWithSuperClass2);
    assertThat(aTypeWithSuperClass).isNotEqualTo(aType);
    assertThat(aTypeWithSuperClass).isEqualTo(aTypeWithSuperClass2);

    ClassSymbolImpl aWithMember = new ClassSymbolImpl("a", "a");
    aWithMember.addMembers(Collections.singleton(new SymbolImpl("fn", "a.fn")));
    RuntimeType aTypeWithMember = new RuntimeType(aWithMember);
    ClassSymbolImpl aWithMember2 = new ClassSymbolImpl("a", "a");
    aWithMember2.addMembers(Collections.singleton(new SymbolImpl("fn", "a.fn")));
    RuntimeType aTypeWithMember2 = new RuntimeType(aWithMember2);
    assertThat(aTypeWithMember).isNotEqualTo(aType);
    assertThat(aTypeWithMember).isEqualTo(aTypeWithMember2);

    RuntimeType x = new RuntimeType(new ClassSymbolImpl("X", null));
    RuntimeType y = new RuntimeType(new ClassSymbolImpl("Y", null));
    assertThat(x).isNotEqualTo(y);
  }

  @Test
  void test_hashCode() {
    RuntimeType aType = new RuntimeType(a);
    assertThat(aType.hashCode()).isEqualTo(new RuntimeType(a).hashCode());
    assertThat(aType.hashCode()).isNotEqualTo(new RuntimeType(b).hashCode());

    RuntimeType x = new RuntimeType(new ClassSymbolImpl("X", null));
    RuntimeType y = new RuntimeType(new ClassSymbolImpl("Y", null));
    assertThat(x.hashCode()).isNotEqualTo(y.hashCode());
  }

  @Test
  void test_toString() {
    assertThat(new RuntimeType(a).toString()).isEqualTo("RuntimeType(a)");
  }

  @Test
  void test_canOnlyBe() {
    assertThat(new RuntimeType(a).canOnlyBe("a")).isTrue();
    assertThat(new RuntimeType(b).canOnlyBe("a")).isFalse();
  }

  @Test
  void test_canBeOrExtend() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    assertThat(new RuntimeType(x).canBeOrExtend("x")).isTrue();
    assertThat(new RuntimeType(x).canBeOrExtend("y")).isFalse();

    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    x2.addSuperClass(x1);
    assertThat(new RuntimeType(x2).canBeOrExtend("x1")).isTrue();

    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    y.addSuperClass(new SymbolImpl("unknown", null));
    assertThat(new RuntimeType(y).canBeOrExtend("z")).isTrue();
  }

  @Test
  void test_isCompatibleWith() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    x2.addSuperClass(x1);

    assertThat(new RuntimeType(x2).isCompatibleWith(new RuntimeType(x1))).isTrue();
    assertThat(new RuntimeType(x1).isCompatibleWith(new RuntimeType(x1))).isTrue();
    assertThat(new RuntimeType(x1).isCompatibleWith(new RuntimeType(x2))).isTrue();

    assertThat(new RuntimeType(x2).isCompatibleWith(new DeclaredType(x1))).isTrue();
    assertThat(new RuntimeType(x1).isCompatibleWith(new DeclaredType(x1))).isTrue();
    assertThat(new RuntimeType(x1).isCompatibleWith(new DeclaredType(x2))).isTrue();
    assertThat(new RuntimeType(x1).isCompatibleWith(new DeclaredType(new SymbolImpl("foo", "foo")))).isTrue();

    ClassSymbolImpl a = new ClassSymbolImpl("a", null);
    ClassSymbolImpl b = new ClassSymbolImpl("b", "b");
    assertThat(new RuntimeType(a).isCompatibleWith(new RuntimeType(b))).isTrue();
    assertThat(new RuntimeType(b).isCompatibleWith(new RuntimeType(a))).isTrue();

    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    ClassSymbolImpl z = new ClassSymbolImpl("z", "z");
    y.addSuperClass(new SymbolImpl("unknown", null));
    assertThat(new RuntimeType(y).isCompatibleWith(new RuntimeType(z))).isTrue();

    FileInput fileInput = PythonTestUtils.parse(
      new SymbolTableBuilder("animals", PythonTestUtils.pythonFile("foo")),
      "class duck:",
      "  def swim(): ...",
      "  def quack(): ...",
      "class goose:",
      "  def swim(): ...");

    ClassDef duckClass = PythonTestUtils.getFirstDescendant(fileInput, tree -> tree.is(Tree.Kind.CLASSDEF));
    ClassDef gooseClass = PythonTestUtils.getLastDescendant(fileInput, tree -> tree.is(Tree.Kind.CLASSDEF));
    ClassSymbol duck = TreeUtils.getClassSymbolFromDef(duckClass);
    ClassSymbol goose = TreeUtils.getClassSymbolFromDef(gooseClass);
    assertThat(new RuntimeType(duck).isCompatibleWith(new RuntimeType(goose))).isTrue();
    assertThat(new RuntimeType(goose).isCompatibleWith(new RuntimeType(duck))).isFalse();
  }

  @Test
  void test_isCompatibleWith_numbers() {
    assertThat(InferredTypes.INT.isCompatibleWith(InferredTypes.INT)).isTrue();
    assertThat(InferredTypes.INT.isCompatibleWith(InferredTypes.FLOAT)).isTrue();
    assertThat(InferredTypes.FLOAT.isCompatibleWith(InferredTypes.INT)).isFalse();
    assertThat(InferredTypes.FLOAT.isCompatibleWith(InferredTypes.COMPLEX)).isTrue();
    assertThat(InferredTypes.FLOAT.isCompatibleWith(InferredTypes.FLOAT)).isTrue();
    assertThat(InferredTypes.INT.isCompatibleWith(InferredTypes.COMPLEX)).isTrue();
    assertThat(InferredTypes.COMPLEX.isCompatibleWith(InferredTypes.COMPLEX)).isTrue();
    assertThat(InferredTypes.COMPLEX.isCompatibleWith(InferredTypes.FLOAT)).isFalse();
    assertThat(InferredTypes.COMPLEX.isCompatibleWith(InferredTypes.INT)).isFalse();
  }

  @Test
  void test_isCompatibleWith_str() {
    InferredType memoryView = runtimeBuiltinType("memoryview");
    assertThat(memoryView.isCompatibleWith(InferredTypes.STR)).isTrue();
    assertThat(InferredTypes.STR.isCompatibleWith(InferredTypes.STR)).isTrue();
    assertThat(InferredTypes.STR.isCompatibleWith(memoryView)).isFalse();
  }

  @Test
  void test_isCompatibleWith_NoneType() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    x1.addMembers(Collections.singletonList(new SymbolImpl("foo", null)));
    ClassSymbolImpl none = new ClassSymbolImpl("NoneType", "NoneType");

    assertThat(new RuntimeType(x1).isCompatibleWith(new RuntimeType(none))).isFalse();
    assertThat(new RuntimeType(none).isCompatibleWith(new RuntimeType(x1))).isFalse();
    assertThat(new RuntimeType(none).isCompatibleWith(new RuntimeType(none))).isTrue();
  }

  @Test
  void test_isCompatibleWith_declared_union_with_missing_symbols() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    ClassSymbolImpl union = new ClassSymbolImpl("union", "typing.Union");
    assertThat(new RuntimeType(x).isCompatibleWith(new DeclaredType(union))).isTrue();
  }

  @Test
  void test_mustBeOrExtend() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    x2.addSuperClass(x1);

    RuntimeType typeX1 = new RuntimeType(x1);
    RuntimeType typeX2 = new RuntimeType(x2);

    assertThat(typeX1.mustBeOrExtend("x1")).isTrue();
    assertThat(typeX1.mustBeOrExtend("x2")).isFalse();

    assertThat(typeX2.mustBeOrExtend("x1")).isTrue();
    assertThat(typeX2.mustBeOrExtend("x2")).isTrue();
  }

  @Test
  void test_resolveDeclaredMember() {
    ClassSymbolImpl typeClass = new ClassSymbolImpl("x", "x");
    SymbolImpl foo = new SymbolImpl("foo", "foo");
    typeClass.addMembers(Collections.singletonList(foo));
    RuntimeType runtimeType = new RuntimeType(typeClass);
    assertThat(runtimeType.resolveDeclaredMember("foo")).contains(foo);
    assertThat(runtimeType.resolveDeclaredMember("bar")).isEmpty();
  }
}
