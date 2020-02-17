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

import static org.assertj.core.api.Assertions.assertThat;

public class RuntimeTypeTest {
  @Test
  public void isIdentityComparableWith() {
    RuntimeType intType = new RuntimeType("int");
    RuntimeType strType = new RuntimeType("str");
    assertThat(intType.isIdentityComparableWith(strType)).isFalse();
    assertThat(intType.isIdentityComparableWith(intType)).isTrue();
    assertThat(intType.isIdentityComparableWith(new RuntimeType("int"))).isTrue();

    assertThat(intType.isIdentityComparableWith(AnyType.ANY)).isTrue();
  }

  @Test
  public void test_equals() {
    RuntimeType intType = new RuntimeType("int");
    assertThat(intType.equals(intType)).isTrue();
    assertThat(intType.equals(new RuntimeType("int"))).isTrue();
    assertThat(intType.equals(new RuntimeType("str"))).isFalse();
    assertThat(intType.equals("int")).isFalse();
    assertThat(intType.equals(null)).isFalse();
  }

  @Test
  public void test_hashCode() {
    RuntimeType intType = new RuntimeType("int");
    assertThat(intType.hashCode()).isEqualTo(new RuntimeType("int").hashCode());
    assertThat(intType.hashCode()).isNotEqualTo(new RuntimeType("str").hashCode());
  }

  @Test
  public void test_toString() {
    assertThat(new RuntimeType("a.b").toString()).isEqualTo("RuntimeType(a.b)");
  }
}
