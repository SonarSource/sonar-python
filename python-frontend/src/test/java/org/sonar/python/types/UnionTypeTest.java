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

import org.junit.Test;
import org.sonar.plugins.python.api.types.InferredType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.UnionType.or;

public class UnionTypeTest {

  private final InferredType a = InferredTypes.runtimeType("a");
  private final InferredType b = InferredTypes.runtimeType("b");
  private final InferredType c = InferredTypes.runtimeType("c");
  private final InferredType d = InferredTypes.runtimeType("d");

  @Test
  public void construction() {
    assertThat(or(anyType(), anyType())).isEqualTo(anyType());
    assertThat(or(anyType(), a)).isEqualTo(anyType());
    assertThat(or(a, anyType())).isEqualTo(anyType());
    assertThat(or(a, a)).isEqualTo(a);
    assertThat(or(a, b)).isNotEqualTo(anyType());
    assertThat(or(or(a, b), c)).isEqualTo(or(a, or(b, c)));
  }

  @Test
  public void isIdentityComparableWith() {
    assertThat(or(a, b).isIdentityComparableWith(anyType())).isTrue();
    assertThat(or(a, b).isIdentityComparableWith(a)).isTrue();
    assertThat(or(a, b).isIdentityComparableWith(b)).isTrue();
    assertThat(or(a, b).isIdentityComparableWith(c)).isFalse();
    assertThat(or(a, b).isIdentityComparableWith(or(b, a))).isTrue();
    assertThat(or(a, b).isIdentityComparableWith(or(c, d))).isFalse();
  }

  @Test
  public void test_equals() {
    assertThat(or(a, b).equals(or(a, b))).isTrue();
    assertThat(or(a, b).equals(or(b, a))).isTrue();
    assertThat(or(a, b).equals(or(a, c))).isFalse();
    assertThat(or(a, b).equals("")).isFalse();
    assertThat(or(a, b).equals(null)).isFalse();
    InferredType aOrB = or(a, b);
    assertThat(aOrB.equals(aOrB)).isTrue();
  }

  @Test
  public void test_hashCode() {
    assertThat(or(a, b).hashCode()).isEqualTo(or(a, b).hashCode());
    assertThat(or(a, b).hashCode()).isNotEqualTo(or(a, c).hashCode());
  }

  @Test
  public void test_toString() {
    assertThat(or(a, b).toString()).isEqualTo("UnionType[RuntimeType(a), RuntimeType(b)]");
  }
}
