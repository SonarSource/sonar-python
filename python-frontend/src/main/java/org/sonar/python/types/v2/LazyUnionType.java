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
package org.sonar.python.types.v2;

import java.util.Collection;
import java.util.Collections;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;

public class LazyUnionType implements PythonType, ResolvableType {

  private final Set<PythonType> candidates;

  private LazyUnionType(Set<PythonType> candidates) {
    this.candidates = candidates;
  }

  @Override
  public PythonType resolve() {
    return UnionType.or(candidates.stream().map(LazyUnionType::resolveCandidate).toList());
  }

  private static PythonType resolveCandidate(PythonType candidate) {
    candidate = TypeUtils.resolved(candidate);
    // ObjectType and SelfType include their wrappers in equals/hashCode. Resolve
    // the complete value structure, including generic arguments, before hashing it.
    // Class members and hierarchies remain lazy; they do not participate in equality.
    if (candidate instanceof ObjectType objectType) {
      return ObjectType.Builder.fromType(objectType)
        .withType(resolveCandidate(objectType.unwrappedType()))
        .withAttributes(objectType.attributes().stream().map(LazyUnionType::resolveCandidate).toList())
        .build();
    }
    if (candidate instanceof SelfType selfType) {
      return SelfType.fromTypeWrapper(TypeWrapper.of(resolveCandidate(selfType.typeWrapper().type())));
    }
    if (candidate instanceof UnionType unionType) {
      return UnionType.or(unionType.candidates().stream().map(LazyUnionType::resolveCandidate).toList());
    }
    return candidate;
  }

  private static Stream<PythonType> flattenLazyUnionTypes(PythonType type) {
    if (type instanceof LazyUnionType lazyUnionType) {
      return lazyUnionType.candidates.stream();
    }
    if (type instanceof UnionType unionType) {
      return unionType.candidates().stream();
    }
    return Stream.of(type);
  }

  Set<PythonType> candidates() {
    return Collections.unmodifiableSet(candidates);
  }

  public static PythonType or(Collection<PythonType> types) {
    types = types.stream().filter(Objects::nonNull).collect(Collectors.toSet());
    if (types.isEmpty()) {
      return PythonType.UNKNOWN;
    }
    if (types.size() == 1) {
      return types.iterator().next();
    }

    Set<PythonType> flatTypes = types.stream().flatMap(LazyUnionType::flattenLazyUnionTypes).collect(Collectors.toSet());
    if (flatTypes.stream().anyMatch(type -> type == PythonType.UNKNOWN)) {
      return PythonType.UNKNOWN;
    }
    flatTypes.remove(PythonType.NEVER);
    if (flatTypes.isEmpty()) {
      return PythonType.NEVER;
    }
    if (flatTypes.size() == 1) {
      return flatTypes.iterator().next();
    }
    return new LazyUnionType(flatTypes);
  }
}
