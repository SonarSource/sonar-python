/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.types;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.runtimeType;
import static org.sonar.python.types.UnionType.or;

class UnionTypeTest {

  private final InferredType a = InferredTypes.runtimeType(new ClassSymbolImpl("a", "a"));
  private final InferredType b = InferredTypes.runtimeType(new ClassSymbolImpl("b", "b"));
  private final InferredType c = InferredTypes.runtimeType(new ClassSymbolImpl("c", "c"));
  private final InferredType d = InferredTypes.runtimeType(new ClassSymbolImpl("d", "d"));

  @Test
  void construction() {
    assertThat(or(anyType(), anyType())).isEqualTo(anyType());
    assertThat(or(anyType(), a)).isEqualTo(anyType());
    assertThat(or(a, anyType())).isEqualTo(anyType());
    assertThat(or(a, a)).isEqualTo(a);
    assertThat(or(a, b)).isNotEqualTo(anyType());
    assertThat(or(or(a, b), c)).isEqualTo(or(a, or(b, c)));
  }

  @Test
  void isIdentityComparableWith() {
    assertThat(or(a, b).isIdentityComparableWith(anyType())).isTrue();
    assertThat(or(a, b).isIdentityComparableWith(a)).isTrue();
    assertThat(or(a, b).isIdentityComparableWith(b)).isTrue();
    assertThat(or(a, b).isIdentityComparableWith(c)).isFalse();
    assertThat(or(a, b).isIdentityComparableWith(or(b, a))).isTrue();
    assertThat(or(a, b).isIdentityComparableWith(or(c, d))).isFalse();
  }

  @Test
  void canHaveMember() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    x.addMembers(Collections.singleton(new SymbolImpl("xxx", null)));
    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    ClassSymbolImpl z = new ClassSymbolImpl("z", "z");
    z.addMembers(Collections.singleton(new SymbolImpl("foo", null)));
    assertThat(or(runtimeType(x), runtimeType(y)).canHaveMember("foo")).isFalse();
    assertThat(or(runtimeType(x), runtimeType(z)).canHaveMember("foo")).isTrue();
  }

  @Test
  void declareMember() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    x.addMembers(Collections.singleton(new SymbolImpl("xxx", null)));
    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    ClassSymbolImpl z = new ClassSymbolImpl("z", "z");
    z.addMembers(Collections.singleton(new SymbolImpl("foo", null)));
    assertThat(or(runtimeType(x), runtimeType(y)).declaresMember("foo")).isFalse();
    assertThat(or(runtimeType(x), runtimeType(z)).declaresMember("foo")).isTrue();
    assertThat(or(new DeclaredType(x), new DeclaredType(z)).declaresMember("foo")).isTrue();
    assertThat(or(new RuntimeType(x), new DeclaredType(y)).declaresMember("xxx")).isTrue();
  }

  @Test
  void resolveMember() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    SymbolImpl foo = new SymbolImpl("foo", null);
    x.addMembers(Arrays.asList(foo, new SymbolImpl("bar", null)));
    ClassSymbolImpl y = new ClassSymbolImpl("y", "y");
    ClassSymbolImpl z = new ClassSymbolImpl("z", "z");
    z.addMembers(Collections.singleton(new SymbolImpl("bar", null)));
    assertThat(or(runtimeType(x), runtimeType(y)).resolveMember("foo")).contains(foo);
    assertThat(or(runtimeType(x), runtimeType(z)).resolveMember("bar")).isEmpty();
    assertThat(or(runtimeType(x), runtimeType(z)).resolveMember("xxx")).isEmpty();

    ClassSymbolImpl classWithUnresolvedHierarchy = new ClassSymbolImpl("u", "u");
    classWithUnresolvedHierarchy.addSuperClass(AmbiguousSymbolImpl.create(new HashSet<>(Arrays.asList(x, new ClassSymbolImpl("x", "x")))));
    assertThat(or(runtimeType(x), runtimeType(classWithUnresolvedHierarchy)).resolveMember("foo")).isEmpty();
  }

  @Test
  void test_equals() {
    assertThat(or(a, b).equals(or(a, b))).isTrue();
    assertThat(or(a, b).equals(or(b, a))).isTrue();
    assertThat(or(a, b).equals(or(a, c))).isFalse();
    assertThat(or(a, b).equals("")).isFalse();
    assertThat(or(a, b).equals(null)).isFalse();
    InferredType aOrB = or(a, b);
    assertThat(aOrB.equals(aOrB)).isTrue();
  }

  @Test
  void test_hashCode() {
    assertThat(or(a, b).hashCode()).isEqualTo(or(a, b).hashCode());
    assertThat(or(a, b).hashCode()).isNotEqualTo(or(a, c).hashCode());
  }

  @Test
  void test_toString() {
    assertThat(or(a, b).toString()).isIn("UnionType[RuntimeType(a), RuntimeType(b)]", "UnionType[RuntimeType(b), RuntimeType(a)]");
  }

  @Test
  void test_canOnlyBe() {
    assertThat(or(a, b).canOnlyBe("a")).isFalse();
    assertThat(or(a, a).canOnlyBe("a")).isTrue();
    assertThat(or(a, a).canOnlyBe("b")).isFalse();
  }

  @Test
  void test_canBeOrExtend() {
    assertThat(or(a, b).canBeOrExtend("a")).isTrue();
    assertThat(or(a, b).canBeOrExtend("c")).isFalse();

    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    x2.addSuperClass(x1);
    assertThat(or(a, new RuntimeType(x2)).canBeOrExtend("x1")).isTrue();
  }

  @Test
  void test_isCompatibleWith() {
    assertThat(a.isCompatibleWith(or(a, b))).isTrue();
    assertThat(or(a, b).isCompatibleWith(a)).isTrue();
    assertThat(c.isCompatibleWith(or(a, b))).isTrue();

    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    ClassSymbolImpl x3 = new ClassSymbolImpl("x3", "x3");
    x2.addSuperClass(x1);
    x2.addMembers(Collections.singleton(new SymbolImpl("foo", null)));
    x3.addMembers(Collections.singleton(new SymbolImpl("bar", null)));
    assertThat(or(new RuntimeType(x2), new RuntimeType(x3)).isCompatibleWith(new RuntimeType(x1))).isTrue();
    assertThat(new RuntimeType(x1).isCompatibleWith(or(new RuntimeType(x2), new RuntimeType(x3)))).isFalse();
  }

  @Test
  void test_isCompatibleWith_NoneType() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    x1.addMembers(Collections.singletonList(new SymbolImpl("foo", null)));
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    x2.addMembers(Collections.singletonList(new SymbolImpl("bar", null)));
    ClassSymbolImpl none = new ClassSymbolImpl("NoneType", "NoneType");

    assertThat(or(new RuntimeType(x1), new RuntimeType(none)).isCompatibleWith(new RuntimeType(none))).isTrue();
    assertThat(or(new RuntimeType(x1), new RuntimeType(none)).isCompatibleWith(new RuntimeType(x1))).isTrue();
    assertThat(new RuntimeType(x1).isCompatibleWith(or(new RuntimeType(x2), new RuntimeType(none)))).isFalse();
    assertThat(or(new RuntimeType(x1), new RuntimeType(none)).isCompatibleWith(or(new RuntimeType(x2), new RuntimeType(none)))).isTrue();
  }

  @Test
  void test_mustBeOrExtend() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    x2.addSuperClass(x1);

    assertThat(or(new RuntimeType(x1), new RuntimeType(x2)).mustBeOrExtend("x1")).isTrue();
    assertThat(or(new RuntimeType(x1), new RuntimeType(x2)).mustBeOrExtend("x2")).isFalse();

    assertThat(or(new RuntimeType(x1), new RuntimeType(x1)).mustBeOrExtend("x1")).isTrue();
  }

  @Test
  void test_resolveDeclaredMember() {
    ClassSymbolImpl typeClassX = new ClassSymbolImpl("x", "x");
    SymbolImpl fooX = new SymbolImpl("foo", "x.foo");
    typeClassX.addMembers(Collections.singletonList(fooX));
    DeclaredType declaredTypeX = new DeclaredType(typeClassX);

    ClassSymbolImpl typeClassY = new ClassSymbolImpl("y", "y");
    SymbolImpl fooY = new SymbolImpl("bar", "y.bar");
    typeClassY.addMembers(Collections.singletonList(fooY));
    DeclaredType declaredTypeY = new DeclaredType(typeClassY);

    InferredType union = or(declaredTypeX, declaredTypeY);
    assertThat(union.resolveDeclaredMember("foo")).contains(fooX);
    assertThat(union.resolveDeclaredMember("baz")).isEmpty();

    ClassSymbolImpl typeClass = new ClassSymbolImpl("x", "x");
    typeClass.addMembers(Collections.singletonList(new SymbolImpl("capitalize", "x.capitalize")));
    InferredType strOrX = or(InferredTypes.STR, new DeclaredType(typeClass));
    assertThat(strOrX.resolveDeclaredMember("capitalize")).isEmpty();

    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    SymbolImpl foo = new SymbolImpl("foo", null);
    x.addMembers(Arrays.asList(foo, new SymbolImpl("bar", null)));
    ClassSymbolImpl classWithUnresolvedHierarchy = new ClassSymbolImpl("u", "u");
    classWithUnresolvedHierarchy.addSuperClass(new SymbolImpl("unresolved", "unresolved"));
    assertThat(or(runtimeType(x), runtimeType(classWithUnresolvedHierarchy)).resolveDeclaredMember("foo")).isEmpty();
    assertThat(or(new DeclaredType(x), new DeclaredType(classWithUnresolvedHierarchy)).resolveDeclaredMember("foo")).isEmpty();
  }
}
