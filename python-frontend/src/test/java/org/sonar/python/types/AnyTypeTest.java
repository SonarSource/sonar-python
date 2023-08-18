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

import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.ClassSymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.AnyType.ANY;

class AnyTypeTest {

  @Test
  void isIdentityComparableWith() {
    assertThat(ANY.isIdentityComparableWith(ANY)).isTrue();
    assertThat(ANY.isIdentityComparableWith(new RuntimeType(new ClassSymbolImpl("a", "a")))).isTrue();
  }

  @Test
  void canHaveMember() {
    assertThat(ANY.canHaveMember("xxx")).isTrue();
  }

  @Test
  void resolveMember() {
    assertThat(ANY.resolveMember("xxx")).isEmpty();
  }

  @Test
  void test_resolveDeclaredMember() {
    assertThat(ANY.resolveDeclaredMember("xxx")).isEmpty();
  }

  @Test
  void test_canOnlyBe() {
    assertThat(ANY.canOnlyBe("a")).isFalse();
  }

  @Test
  void test_canBeOrExtend() {
    assertThat(ANY.canBeOrExtend("a")).isTrue();
    assertThat(InferredTypes.INT.isCompatibleWith(ANY)).isTrue();
    assertThat(ANY.isCompatibleWith(InferredTypes.INT)).isTrue();
  }

  @Test
  void test_mustBeOrExtend() {
    assertThat(ANY.mustBeOrExtend("a")).isFalse();
  }

  @Test
  void test_declaresMember() {
    assertThat(ANY.declaresMember("foo")).isTrue();
  }
}
