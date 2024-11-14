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

import java.util.Map;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

class ModuleTypeTest {


  @Test
  void resolveMemberTest() {
    var a = new ModuleType("a");
    var b = new ModuleType("b");
    b.members().put("a", TypeWrapper.of(a));

    Assertions.assertThat(b.resolveMember("a")).containsSame(a);
    Assertions.assertThat(b.resolveMember("b")).isNotPresent();
    Assertions.assertThat(b.unwrappedType()).isEqualTo(b);
  }

  @Test
  void replaceMemberIfUnknown() {
    var a = new ModuleType("a");
    a.members().put("b", TypeWrapper.UNKNOWN_TYPE_WRAPPER);
    Assertions.assertThat(a.resolveMember("b")).containsSame(PythonType.UNKNOWN);
    var b = new ModuleType("b", a);
    Assertions.assertThat(a.resolveMember("b")).containsSame(PythonType.UNKNOWN);
    Assertions.assertThat(a.resolveSubmodule("b")).containsSame(b);
  }

  @Test
  void doNotReplaceKnownMember() {
    var a = new ModuleType("a");
    ClassType existingMember = new ClassType("b", "mymod.b");
    a.members().put("b", TypeWrapper.of(existingMember));
    Assertions.assertThat(a.resolveMember("b")).containsSame(existingMember);
    new ModuleType("b", a);
    Assertions.assertThat(a.resolveMember("b")).containsSame(existingMember);
  }

  @Test
  void registerAsSubmoduleTest() {
    var a = new ModuleType("a");
    a.members().put("b", TypeWrapper.UNKNOWN_TYPE_WRAPPER);
    Assertions.assertThat(a.resolveSubmodule("b")).isEmpty();
    var b = new ModuleType("b", a, Map.of());
    Assertions.assertThat(a.resolveMember("b")).containsSame(PythonType.UNKNOWN);
    Assertions.assertThat(a.resolveSubmodule("b")).containsSame(b);
    // no parent: no effect
    new ModuleType("c", null, Map.of());
  }

  @Test
  void doNotReplaceKnownSubmodule() {
    var a = new ModuleType("a");
    ModuleType existingSubmodule = new ModuleType("b", a, Map.of());
    Assertions.assertThat(a.resolveSubmodule("b")).containsSame(existingSubmodule);
    new ModuleType("b", a, Map.of());
    Assertions.assertThat(a.resolveSubmodule("b")).containsSame(existingSubmodule);
  }

  @Test
  void toStringTest() {
    var root = new ModuleType(null);
    var module = new ModuleType("pkg", root);
    var moduleString = module.toString();
    Assertions.assertThat(moduleString).isEqualTo("ModuleType{name='pkg', members={}}");
  }
}
