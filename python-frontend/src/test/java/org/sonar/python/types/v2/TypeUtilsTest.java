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


import java.util.Set;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.TypesTestUtils.BOOL_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.FLOAT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.NONE_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;

class TypeUtilsTest {
  @Test
  void testGetNestedEffectiveTypes() {
    assertThat(TypeUtils.getNestedEffectiveTypes(INT_TYPE)).containsExactlyInAnyOrder(INT_TYPE);

    assertThat(TypeUtils.getNestedEffectiveTypes(UnionType.or(INT_TYPE, FLOAT_TYPE))).containsExactlyInAnyOrder(INT_TYPE, FLOAT_TYPE);
    assertThat(TypeUtils.getNestedEffectiveTypes(new UnionType(Set.of(INT_TYPE, UnionType.or(FLOAT_TYPE, BOOL_TYPE))))).containsExactlyInAnyOrder(INT_TYPE, FLOAT_TYPE, BOOL_TYPE);

    assertThat(TypeUtils.getNestedEffectiveTypes(new LazyUnionType(Set.of(INT_TYPE, FLOAT_TYPE)))).containsExactlyInAnyOrder(INT_TYPE, FLOAT_TYPE);
    assertThat(TypeUtils.getNestedEffectiveTypes(
      new UnionType(Set.of(
        new LazyUnionType(Set.of(INT_TYPE, FLOAT_TYPE)),
        new LazyUnionType(Set.of(BOOL_TYPE))
      )))).containsExactlyInAnyOrder(INT_TYPE, FLOAT_TYPE, BOOL_TYPE);

    assertThat(TypeUtils.getNestedEffectiveTypes(
      PROJECT_LEVEL_TYPE_TABLE.lazyTypesContext().getOrCreateLazyType("int")
    )).containsExactlyInAnyOrder(INT_TYPE);

    assertThat(TypeUtils.getNestedEffectiveTypes(
      new UnionType(Set.of(
        PROJECT_LEVEL_TYPE_TABLE.lazyTypesContext().getOrCreateLazyType("int"),
        new LazyUnionType(Set.of(FLOAT_TYPE, BOOL_TYPE))
      ))
    )).containsExactlyInAnyOrder(INT_TYPE, FLOAT_TYPE, BOOL_TYPE);
  }

  @Test
  void testUnwrapType() {
    var intObjType = new ObjectType(INT_TYPE);
    var floatObjType = new ObjectType(FLOAT_TYPE);
    assertThat(TypeUtils.unwrapType(INT_TYPE)).isEqualTo(INT_TYPE);
    assertThat(TypeUtils.unwrapType(intObjType)).isEqualTo(INT_TYPE);
    assertThat(TypeUtils.unwrapType(new ObjectType(new ObjectType(INT_TYPE)))).isEqualTo(INT_TYPE);

    assertThat(TypeUtils.unwrapType(UnionType.or(intObjType, FLOAT_TYPE))).isEqualTo(UnionType.or(INT_TYPE, FLOAT_TYPE));

    var nestedUnions = UnionType.or(
      intObjType,
      UnionType.or(
        floatObjType,
        NONE_TYPE
      )
    );

    assertThat(TypeUtils.unwrapType(nestedUnions))
      .isEqualTo(
        UnionType.or(INT_TYPE, UnionType.or(FLOAT_TYPE, NONE_TYPE))
      );
  }

}