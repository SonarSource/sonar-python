/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.api.types.v2.ClassTypeTest.classTypes;

class SuperProxyTypeTest {

  @Test
  void resolves_only_the_direct_local_member_with_unresolved_distant_ancestry() {
    var classes = classTypes("""
      class Base(unknown.Ancestor):
        def target(self): ...
      class Child(Base): ...
      """);
    var base = classes.getFirst();
    var owner = classes.getLast();
    var proxy = new SuperProxyType(owner);
    assertThat(owner.mro()).isEmpty();
    assertThat(base.mro()).isEmpty();
    assertThat(proxy.resolveMember("target")).isEqualTo(base.resolveMember("target"));
    assertThat(proxy.resolveMember("missing")).isEmpty();
  }

  @Test
  void does_not_search_beyond_the_direct_base_with_unresolved_ancestry() {
    var classes = classTypes("""
      class Root(unknown.Ancestor):
        target = 42
      class Base(Root): ...
      class Child(Base): ...
      """);
    assertThat(new SuperProxyType(classes.getLast()).resolveMember("target")).isEmpty();
  }

  @Test
  void does_not_guess_between_multiple_bases_in_an_incomplete_hierarchy() {
    var classes = classTypes("""
      class Base(unknown.Ancestor):
        target = 42
      class Other: ...
      class Child(Base, Other): ...
      """);
    assertThat(new SuperProxyType(classes.getLast()).resolveMember("target")).isEmpty();
  }

  @Test
  void self_inheritance_does_not_resolve_members() {
    var owner = new ClassType("Owner", "Owner");
    owner.superClasses().add(TypeWrapper.of(owner));
    owner.members().add(new Member("target", PythonType.UNKNOWN));
    assertThat(new SuperProxyType(owner).resolveMember("target")).isEmpty();
  }

  @Test
  void cyclic_inheritance_with_unknown_ancestry_does_not_resolve_members() {
    var owner = new ClassType("Owner", "Owner");
    var base = new ClassType("Base", "Base");
    owner.superClasses().add(TypeWrapper.of(base));
    base.superClasses().add(TypeWrapper.of(owner));
    base.superClasses().add(TypeWrapper.UNKNOWN_TYPE_WRAPPER);
    base.members().add(new Member("target", PythonType.UNKNOWN));
    assertThat(new SuperProxyType(owner).resolveMember("target")).isEmpty();
  }

  @Test
  void distant_cycle_with_unknown_ancestry_does_not_resolve_members() {
    var owner = new ClassType("Owner", "Owner");
    var base = new ClassType("Base", "Base");
    owner.superClasses().add(TypeWrapper.of(base));
    base.superClasses().add(TypeWrapper.of(base));
    base.superClasses().add(TypeWrapper.UNKNOWN_TYPE_WRAPPER);
    base.members().add(new Member("target", PythonType.UNKNOWN));
    assertThat(new SuperProxyType(owner).resolveMember("target")).isEmpty();
  }

  @Test
  void conflicting_known_ancestry_with_unknown_base_does_not_resolve_members() {
    var classes = classTypes("""
      class Root: ...
      class Left(Root): ...
      class Base(Root, Left, unknown.Ancestor):
        target = 42
      class Child(Base): ...
      """);
    assertThat(new SuperProxyType(classes.getLast()).resolveMember("target")).isEmpty();
  }
}
