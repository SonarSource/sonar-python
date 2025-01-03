/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.types;

import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;

import static org.sonar.python.types.InferredTypes.anyType;

class UnionType implements InferredType {

  private final Set<InferredType> types;

  private UnionType(Set<InferredType> types) {
    this.types = types;
  }

  public static InferredType or(InferredType type1, InferredType type2) {
    if (type1.equals(anyType()) || type2.equals(anyType())) {
      return anyType();
    }
    if (type1.equals(type2)) {
      return type1;
    }
    Set<InferredType> types = new HashSet<>();
    addTypes(type1, types);
    addTypes(type2, types);
    return new UnionType(types);
  }

  private static void addTypes(InferredType type, Set<InferredType> types) {
    if (type instanceof UnionType unionType) {
      types.addAll(unionType.types);
    } else {
      types.add(type);
    }
  }

  @Override
  public boolean isIdentityComparableWith(InferredType other) {
    return types.stream().anyMatch(t -> t.isIdentityComparableWith(other));
  }

  @Override
  public boolean canHaveMember(String memberName) {
    return types.stream().anyMatch(t -> t.canHaveMember(memberName));
  }

  @Override
  public boolean declaresMember(String memberName) {
    return types.stream().anyMatch(t -> t.declaresMember(memberName));
  }

  @Override
  public Optional<Symbol> resolveMember(String memberName) {
    if (hasUnresolvedHierarchy()) {
      return Optional.empty();
    }
    Set<Optional<Symbol>> resolved = types.stream()
      .map(t -> t.resolveMember(memberName))
      .filter(Optional::isPresent)
      .collect(Collectors.toSet());
    return resolved.size() == 1 ? resolved.iterator().next() : Optional.empty();
  }

  @Override
  public Optional<Symbol> resolveDeclaredMember(String memberName) {
    if (hasUnresolvedHierarchy()) {
      return Optional.empty();
    }
    Set<Optional<Symbol>> resolved = types.stream()
      .map(t -> t.resolveDeclaredMember(memberName))
      .filter(Optional::isPresent)
      .collect(Collectors.toSet());
    return resolved.size() == 1 ? resolved.iterator().next() : Optional.empty();
  }

  private boolean hasUnresolvedHierarchy() {
    for (InferredType type : types) {
      if (type instanceof RuntimeType runtimeType && runtimeType.hasUnresolvedHierarchy()) {
        return true;
      }
      if (type instanceof DeclaredType declaredType && declaredType.hasUnresolvedHierarchy()) {
        return true;
      }
    }
    return false;
  }

  @Override
  public boolean canOnlyBe(String typeName) {
    return types.stream().allMatch(t -> t.canOnlyBe(typeName));
  }

  @Override
  public boolean canBeOrExtend(String typeName) {
    return types.stream().anyMatch(t -> t.canBeOrExtend(typeName));
  }

  @Override
  public boolean isCompatibleWith(InferredType other) {
    return types.stream().anyMatch(t -> t.isCompatibleWith(other));
  }

  public boolean mustBeOrExtend(String fullyQualifiedName) {
    return types.stream().allMatch(t -> t.mustBeOrExtend(fullyQualifiedName));
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    UnionType unionType = (UnionType) o;
    return Objects.equals(types, unionType.types);
  }

  @Override
  public int hashCode() {
    return Objects.hash(types);
  }

  @Override
  public String toString() {
    return "UnionType" + types;
  }

  Set<InferredType> types() {
    return Collections.unmodifiableSet(types);
  }
}
