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
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.or;
import static org.sonar.python.types.InferredTypes.runtimeType;

public class InferredTypesTest {

  @Test
  public void test_runtimeType() {
    assertThat(runtimeType(null)).isEqualTo(anyType());
    assertThat(runtimeType("a.b")).isEqualTo(new RuntimeType("a.b"));
  }

  @Test
  public void test_or() {
    assertThat(or(anyType(), anyType())).isEqualTo(anyType());
    assertThat(or(anyType(), runtimeType("a"))).isEqualTo(anyType());
    assertThat(or(runtimeType("a"), anyType())).isEqualTo(anyType());
    assertThat(or(runtimeType("a"), runtimeType("a"))).isEqualTo(runtimeType("a"));
    assertThat(or(runtimeType("a"), runtimeType("b"))).isNotEqualTo(anyType());
    assertThat(or(runtimeType("a"), runtimeType("b"))).isEqualTo(or(runtimeType("b"), runtimeType("a")));
  }
}
