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
import org.sonar.plugins.python.api.symbols.ClassSymbol;

import static org.assertj.core.api.Assertions.assertThat;

public class TypeShedTest {

  @Test
  public void classes() {
    ClassSymbol intClass = TypeShed.typeShedClass("int");
    assertThat(intClass.superClasses()).isEmpty();
    assertThat(intClass.hasUnresolvedTypeHierarchy()).isFalse();

    assertThat(TypeShed.typeShedClass("bool").superClasses()).containsExactly(intClass);
  }

  @Test(expected = IllegalArgumentException.class)
  public void not_a_class() {
    TypeShed.typeShedClass("repr");
  }

  @Test(expected = IllegalArgumentException.class)
  public void unknown_name() {
    TypeShed.typeShedClass("xxx");
  }

  @Test
  public void none_type() {
    ClassSymbol noneType = TypeShed.typeShedClass("NoneType");
    assertThat(noneType.superClasses()).isEmpty();
  }
}
