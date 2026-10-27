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

import java.util.Optional;

/**
 * A proxy type returned by zero-argument {@code super()} calls inside a method.
 * Member lookup delegates to the enclosing class's C3 MRO tail — i.e. the MRO
 * starting from the first parent (skipping the class itself).
 * Known class-base alternatives are checked through every valid C3 order.
 * With unresolved distant ancestry, only a sole known direct parent's local members
 * can be resolved. As with complete MRO lookup, this models the enclosing class;
 * runtime changes caused by subclassing or class decorators are not modeled.
 */
public record SuperProxyType(ClassType ownerClassType) implements PythonType {

  // FQN used by the V2 type system for the builtin super class
  // (builtins prefix is stripped by TypeShedUtils.normalizedFqn)
  public static final String SUPER_FQN = "super";

  @Override
  public String name() {
    return SUPER_FQN;
  }

  public boolean canResolveMembers() {
    return ownerClassType.mro().filter(mro -> mro.size() > 1).isPresent()
      || ClassTypeMroUtils.computeAlternatives(ownerClassType).isPresent()
      || directBaseInIncompleteHierarchy().isPresent();
  }

  private Optional<ClassType> directBaseInIncompleteHierarchy() {
    var bases = ownerClassType.superClasses();
    if (bases.size() == 1 && bases.getFirst().type() instanceof ClassType base && ownerClassType.hasUnresolvedHierarchy()
      // Validate known ancestry without using its partial linearization for member lookup.
      && ClassTypeMroUtils.compute(ownerClassType).isPresent()) {
      return Optional.of(base);
    }
    return Optional.empty();
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return ownerClassType.mro()
      .map(mro -> mro.stream()
        .skip(1)
        .map(classType -> classType.localMember(memberName))
        .filter(Optional::isPresent)
        .map(Optional::get)
        .findFirst())
      .orElseGet(() -> ClassTypeMroUtils.computeAlternatives(ownerClassType)
        .map(orders -> ClassTypeMroUtils.inheritedMember(orders, memberName))
        .orElseGet(() -> directBaseInIncompleteHierarchy().flatMap(base -> base.localMember(memberName))));
  }
}
