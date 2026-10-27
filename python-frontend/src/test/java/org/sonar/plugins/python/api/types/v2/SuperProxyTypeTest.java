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

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.v2.ClassTypeBuilder;

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

  @Test
  void resolves_inherited_contract_across_versioned_diamond_bases() {
    var earlier = new ClassType("Root", "library.Root");
    var later = new ClassType("Root", "library.Root");
    var roots = UnionType.or(earlier, later);
    var publicRoot = new ClassTypeBuilder("PublicRoot", "library.PublicRoot").addSuperClass(roots).build();
    var privateBuffer = new ClassTypeBuilder("PrivateBuffer", "library.PrivateBuffer").addSuperClass(roots).build();
    var target = classTypes("class Signature:\n  def target(self, value=0, /): ...").getFirst().resolveMember("target").orElseThrow();
    privateBuffer.members().add(new Member("target", target));
    var buffer = new ClassTypeBuilder("Buffer", "library.Buffer").withSuperClasses(privateBuffer, publicRoot).build();
    var child = new ClassTypeBuilder("Child", "Child").addSuperClass(buffer).build();
    var proxy = new SuperProxyType(child);

    // No one concrete order represents both Python versions. The inherited
    // signature is nevertheless selected in every valid C3 linearization.
    assertThat(child.mro()).isEmpty();
    assertThat(proxy.canResolveMembers()).isTrue();
    assertThat(proxy.resolveMember("target")).contains(target);
    assertThat(proxy.resolveMember("missing")).isEmpty();
  }


  @Test
  void versioned_diamonds_preserve_sibling_override_precedence() {
    var earlier = new ClassType("Root", "library.Root");
    var later = new ClassType("Root", "library.Root");
    earlier.members().add(new Member("target", earlier));
    later.members().add(new Member("target", later));
    var roots = UnionType.or(earlier, later);
    var left = new ClassTypeBuilder("Left", "Left").addSuperClass(roots).build();
    var right = new ClassTypeBuilder("Right", "Right").addSuperClass(roots).build();
    right.members().add(new Member("target", right));
    var owner = new ClassTypeBuilder("Owner", "Owner").withSuperClasses(left, right).build();

    assertThat(ClassTypeMroUtils.computeAlternatives(owner)).hasValueSatisfying(orders -> {
      assertThat(orders).hasSize(2);
      assertThat(orders).allSatisfy(order -> assertThat(order.subList(0, 3)).containsExactly(owner, left, right));
    });
    assertThat(new SuperProxyType(owner).canResolveMembers()).isTrue();
    assertThat(new SuperProxyType(owner).resolveMember("target")).contains(right);
    assertThat(owner.resolveMember("target")).contains(right);
  }

  @Test
  void preserves_different_members_across_alternative_orders() {
    var earlier = new ClassType("Root", "library.Root");
    var later = new ClassType("Root", "library.Root");
    earlier.members().add(new Member("target", earlier));
    later.members().add(new Member("target", later));
    var owner = new ClassTypeBuilder("Owner", "Owner").addSuperClass(UnionType.or(earlier, later)).build();

    assertThat(new SuperProxyType(owner).resolveMember("target")).hasValueSatisfying(member ->
      assertThat(member).isInstanceOfSatisfying(UnionType.class,
        union -> assertThat(union.candidates()).containsExactlyInAnyOrder(earlier, later)));
  }

  @Test
  void member_missing_from_one_alternative_stays_unknown() {
    var earlier = new ClassType("Root", "library.Root");
    var later = new ClassType("Root", "library.Root");
    earlier.members().add(new Member("target", earlier));
    var owner = new ClassTypeBuilder("Owner", "Owner").addSuperClass(UnionType.or(earlier, later)).build();

    assertThat(new SuperProxyType(owner).resolveMember("target")).contains(PythonType.UNKNOWN);
    assertThat(owner.resolveMember("target")).contains(PythonType.UNKNOWN);
    assertThat(new SuperProxyType(owner).resolveMember("missing")).isEmpty();
  }

  @Test
  void any_unknown_alternative_ancestry_prevents_inherited_lookup() {
    var earlier = new ClassType("Root", "library.Root");
    var later = new ClassTypeBuilder("Root", "library.Root").addSuperClass(PythonType.UNKNOWN).build();
    var parent = new ClassTypeBuilder("Parent", "Parent").addSuperClass(UnionType.or(earlier, later)).build();
    parent.members().add(new Member("target", parent));
    var bridge = new ClassTypeBuilder("Bridge", "Bridge").addSuperClass(parent).build();
    var owner = new ClassTypeBuilder("Owner", "Owner").addSuperClass(bridge).build();

    assertThat(ClassTypeMroUtils.computeAlternatives(owner)).isEmpty();
    assertThat(new SuperProxyType(owner).resolveMember("target")).isEmpty();
  }

  @Test
  void any_invalid_alternative_order_prevents_inherited_lookup() {
    var root = new ClassType("Root", "Root");
    var other = new ClassType("Other", "Other");
    var leaf = new ClassTypeBuilder("Leaf", "Leaf").addSuperClass(root).build();
    leaf.members().add(new Member("target", leaf));
    // (Other, Leaf) is valid, but (Root, Leaf) violates C3.
    var parent = new ClassTypeBuilder("Parent", "Parent").withSuperClasses(UnionType.or(root, other), leaf).build();
    var owner = new ClassTypeBuilder("Owner", "Owner").addSuperClass(parent).build();

    assertThat(ClassTypeMroUtils.computeAlternatives(owner)).isEmpty();
    assertThat(new SuperProxyType(owner).resolveMember("target")).isEmpty();
  }

  @Test
  void any_cyclic_alternative_prevents_inherited_lookup() {
    var earlier = new ClassType("Root", "library.Root");
    var later = new ClassType("Root", "library.Root");
    later.superClasses().add(TypeWrapper.of(later));
    var parent = new ClassTypeBuilder("Parent", "Parent").addSuperClass(UnionType.or(earlier, later)).build();
    parent.members().add(new Member("target", parent));
    var bridge = new ClassTypeBuilder("Bridge", "Bridge").addSuperClass(parent).build();
    var owner = new ClassTypeBuilder("Owner", "Owner").addSuperClass(bridge).build();

    assertThat(ClassTypeMroUtils.computeAlternatives(owner)).isEmpty();
    assertThat(new SuperProxyType(owner).resolveMember("target")).isEmpty();
  }

  @Test
  void non_class_base_alternatives_are_unsupported() {
    var parent = new ClassType("Parent", "Parent");
    var function = classTypes("class Signature:\n  def target(self): ...").getFirst().resolveMember("target").orElseThrow();
    var owner = new ClassTypeBuilder("Owner", "Owner").addSuperClass(UnionType.or(parent, function)).build();

    assertThat(ClassTypeMroUtils.computeAlternatives(owner)).isEmpty();
    assertThat(new SuperProxyType(owner).canResolveMembers()).isFalse();
    assertThat(new SuperProxyType(owner).resolveMember("target")).isEmpty();
  }

  @Test
  void limits_alternative_combinations() {
    var root = new ClassType("Root", "Root");
    root.members().add(new Member("target", root));
    ClassType owner = root;
    for (int index = 0; index < 5; index++) {
      var choice = UnionType.or(new ClassType("Earlier" + index, "Earlier" + index), new ClassType("Later" + index, "Later" + index));
      owner = new ClassTypeBuilder("Layer" + index, "Layer" + index).withSuperClasses(owner, choice).build();
    }
    assertThat(ClassTypeMroUtils.computeAlternatives(owner)).hasValueSatisfying(orders -> assertThat(orders).hasSize(32));
    assertThat(new SuperProxyType(owner).resolveMember("target")).contains(root);

    var tooMany = new ClassTypeBuilder("TooMany", "TooMany")
      .withSuperClasses(owner, UnionType.or(new ClassType("Earlier", "Earlier"), new ClassType("Later", "Later"))).build();
    assertThat(ClassTypeMroUtils.computeAlternatives(tooMany)).isEmpty();
    assertThat(new SuperProxyType(tooMany).canResolveMembers()).isFalse();
    assertThat(new SuperProxyType(tooMany).resolveMember("target")).isEmpty();
  }


  @Test
  void snapshots_base_wrappers_whose_resolution_creates_fresh_unions() {
    var earlier = new ClassType("Root", "library.Root");
    var later = new ClassType("Root", "library.Root");
    earlier.members().add(new Member("target", earlier));
    later.members().add(new Member("target", later));
    var reads = new AtomicInteger();
    TypeWrapper deferred = () -> {
      reads.incrementAndGet();
      return UnionType.or(earlier, later);
    };
    var left = new ClassTypeBuilder("Left", "Left").addSuperClass(deferred).build();
    var right = new ClassTypeBuilder("Right", "Right").addSuperClass(deferred).build();
    var owner = new ClassTypeBuilder("Owner", "Owner").withSuperClasses(left, right).build();

    assertThat(ClassTypeMroUtils.computeAlternatives(owner)).hasValueSatisfying(orders -> assertThat(orders).hasSize(2));
    assertThat(reads).hasValue(1);
    assertThat(new SuperProxyType(owner).resolveMember("target")).hasValueSatisfying(member ->
      assertThat(member).isInstanceOfSatisfying(UnionType.class,
        union -> assertThat(union.candidates()).containsExactlyInAnyOrder(earlier, later)));
  }

}
