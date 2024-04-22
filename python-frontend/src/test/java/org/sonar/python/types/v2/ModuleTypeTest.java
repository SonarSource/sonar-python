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

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

class ModuleTypeTest {


  @Test
  void resolveMemberTest() {
    var a = new ModuleType("a");
    var b = new ModuleType("b");
    b.members().put("a", a);

    var resolved = b.resolveMember("a");
    Assertions.assertThat(resolved).isSameAs(a);

    resolved = b.resolveMember("b");
    Assertions.assertThat(resolved).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void toStringTest() {
    var root = new ModuleType(null);
    var module = new ModuleType("pkg", root);
    var moduleString = module.toString();
    Assertions.assertThat(moduleString).isEqualTo("ModuleType{name='pkg', members={}}");
  }

  @Test
  void equalsTest() {
    var parent1 = new ModuleType(null);
    var module1 = new ModuleType("a", parent1);
    parent1.members().put("a", module1);

    var parent2 = new ModuleType(null);
    var module2 = new ModuleType("a", parent2);
    parent2.members().put("a", module2);

    var module3 = module1;

    var module4 = new ModuleType("b");

    var module5 = new ModuleType("a");
    module5.members().put("b", module4);

    Assertions.assertThat(module1).isEqualTo(module2)
      .isEqualTo(module3)
      .isNotEqualTo(module4)
      .isNotEqualTo(module5);
  }

  @Test
  void hashCodeTest() {
    var module1 = new ModuleType("a");
    var module2 = new ModuleType("a");
    var module3 = new ModuleType("b");

    Assertions.assertThat(module1)
      .hasSameHashCodeAs(module2)
      .doesNotHaveSameHashCodeAs(module3);
  }



}
