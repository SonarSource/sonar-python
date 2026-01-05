/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.plugins.python.api.types.v2;

import java.util.HashMap;
import java.util.Map;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

class ModuleTypeTest {

  @Test
  void constructorTest() {
    var a = new ModuleType("a");
    var b = new ModuleType("b", "a.b", a, new HashMap<>());

    Assertions.assertThat(b.fullyQualifiedName()).isEqualTo("a.b");
    Assertions.assertThat(a.resolveMember("b")).isEmpty();
  }

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
    a.registerSubmodule(b);
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
    a.registerSubmodule(b);
    Assertions.assertThat(a.resolveMember("b")).containsSame(PythonType.UNKNOWN);
    Assertions.assertThat(a.resolveSubmodule("b")).containsSame(b);
    // no parent: no effect
    new ModuleType("c", null, Map.of());
  }

  @Test
  void doNotReplaceKnownSubmodule() {
    var a = new ModuleType("a");
    ModuleType existingSubmodule = new ModuleType("b", a, Map.of());
    a.registerSubmodule(existingSubmodule);
    Assertions.assertThat(a.resolveSubmodule("b")).containsSame(existingSubmodule);
    var b = new ModuleType("b", a, Map.of());
    a.registerSubmodule(b);
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
