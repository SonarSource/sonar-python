/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;

import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.InferredTypes.or;

public class DeclaredTypeTest {

  private final ClassSymbolImpl a = new ClassSymbolImpl("a", "a");
  private final ClassSymbolImpl b = new ClassSymbolImpl("b", "b");
  private final ClassSymbolImpl c = new ClassSymbolImpl("c", "c");

  @Test
  public void isIdentityComparableWith() {
    DeclaredType aType = new DeclaredType(a);
    DeclaredType bType = new DeclaredType(b);
    DeclaredType cType = new DeclaredType(c);

    assertThat(aType.isIdentityComparableWith(bType)).isTrue();
    assertThat(aType.isIdentityComparableWith(aType)).isTrue();
    assertThat(aType.isIdentityComparableWith(new RuntimeType(a))).isTrue();

    assertThat(aType.isIdentityComparableWith(AnyType.ANY)).isTrue();

    assertThat(aType.isIdentityComparableWith(or(aType, bType))).isTrue();
    assertThat(aType.isIdentityComparableWith(or(cType, bType))).isTrue();
  }

  @Test
  public void member() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    SymbolImpl foo = new SymbolImpl("foo", null);
    x.addMembers(singletonList(foo));
    assertThat(new DeclaredType(x).canHaveMember("foo")).isTrue();
    assertThat(new DeclaredType(x).canHaveMember("bar")).isTrue();
    assertThat(new DeclaredType(x).resolveMember("foo")).isEmpty();
    assertThat(new DeclaredType(x).resolveMember("bar")).isEmpty();
  }

  @Test
  public void test_toString() {
    assertThat(new DeclaredType(a)).hasToString("DeclaredType(a)");
    assertThat(new DeclaredType(a, Collections.singletonList(new DeclaredType(b)))).hasToString("DeclaredType(a[b])");
    assertThat(new DeclaredType(a, Arrays.asList(new DeclaredType(b), new DeclaredType(c)))).hasToString("DeclaredType(a[b, c])");
  }

  @Test
  public void test_canOnlyBe() {
    assertThat(new DeclaredType(a).canOnlyBe("a")).isFalse();
    assertThat(new DeclaredType(b).canOnlyBe("a")).isFalse();
  }

  @Test
  public void test_canBeOrExtend() {
    ClassSymbolImpl x = new ClassSymbolImpl("x", "x");
    assertThat(new DeclaredType(x).canBeOrExtend("x")).isTrue();
    assertThat(new DeclaredType(x).canBeOrExtend("y")).isTrue();
  }

  @Test
  public void test_isCompatibleWith() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    x2.addSuperClass(x1);

    assertThat(new DeclaredType(x2).isCompatibleWith(new DeclaredType(x1))).isTrue();
    assertThat(new DeclaredType(x1).isCompatibleWith(new DeclaredType(x1))).isTrue();
    assertThat(new DeclaredType(x1).isCompatibleWith(new DeclaredType(x2))).isTrue();
  }

  @Test
  public void test_mustBeOrExtend() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    ClassSymbolImpl x2 = new ClassSymbolImpl("x2", "x2");
    x2.addSuperClass(x1);

    DeclaredType typeX1 = new DeclaredType(x1);
    DeclaredType typeX2 = new DeclaredType(x2);

    assertThat(typeX1.mustBeOrExtend("x1")).isTrue();
    assertThat(typeX1.mustBeOrExtend("x2")).isFalse();

    assertThat(typeX2.mustBeOrExtend("x1")).isTrue();
    assertThat(typeX2.mustBeOrExtend("x2")).isTrue();

    ClassSymbolImpl otherX1 = new ClassSymbolImpl("x1", "x1");
    Set<Symbol> symbols = new HashSet<>(Arrays.asList(x1, otherX1));
    AmbiguousSymbol ambiguousSymbol = new AmbiguousSymbolImpl("x1", "x1", symbols);
    DeclaredType typeAmbiguousX1 = new DeclaredType(ambiguousSymbol);
    assertThat(typeAmbiguousX1.mustBeOrExtend("x1")).isTrue();
    assertThat(typeAmbiguousX1.mustBeOrExtend("other")).isFalse();

    DeclaredType declaredType = new DeclaredType(new SymbolImpl("C", "foo.C"));
    assertThat(declaredType.mustBeOrExtend("other")).isTrue();
    assertThat(declaredType.mustBeOrExtend("foo.C")).isTrue();
    assertThat(declaredType.mustBeOrExtend("C")).isTrue();
  }

  @Test
  public void test_getClass() {
    ClassSymbolImpl x1 = new ClassSymbolImpl("x1", "x1");
    assertThat(new DeclaredType(x1).getTypeClass()).isEqualTo(x1);
  }
}
