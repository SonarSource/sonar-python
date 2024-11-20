/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.types.v2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.api.Beta;

@Beta
public record UnionType(Set<PythonType> candidates) implements PythonType {

  @Override
  public Optional<String> displayName() {
    List<String> candidateNames = new ArrayList<>();
    for (PythonType candidate : candidates) {
      Optional<String> s = candidate.displayName();
      s.ifPresent(candidateNames::add);
      if (s.isEmpty()) {
        return Optional.empty();
      }
    }
    String name = candidateNames.stream().sorted().collect(Collectors.joining(", ", "Union[", "]"));
    return Optional.of(name);
  }


  /**
   * For UnionType, hasMember will return true if all alternatives have the member
   * It will return false if all alternatives DON'T have the member
   * It will return unknown in all other cases
   */
  @Override
  public TriBool hasMember(String memberName) {
    Set<TriBool> uniqueResult = candidates.stream().map(c -> c.hasMember(memberName)).collect(Collectors.toSet());
    return uniqueResult.size() == 1 ? uniqueResult.iterator().next() : TriBool.UNKNOWN;
  }

  @Override
  public boolean isCompatibleWith(PythonType another) {
    return candidates.isEmpty() || candidates.stream()
      .anyMatch(candidate -> candidate.isCompatibleWith(another));
  }

  @Override
  public TypeSource typeSource() {
    return candidates.stream().map(PythonType::typeSource)
      .min(Comparator.comparing(TypeSource::score))
      .orElse(TypeSource.EXACT);
  }

  @Beta
  public static PythonType or(Collection<PythonType> candidates) {
    ensureCandidatesAreNotLazyTypes(candidates);
    if (candidates.isEmpty()) {
      return PythonType.UNKNOWN;
    }
    return candidates
      .stream()
      .reduce(new UnionType(new HashSet<>()), UnionType::or);
  }

  @Beta
  public static PythonType or(@Nullable PythonType type1, @Nullable PythonType type2) {
    if (type1 == null) {
      return type2;
    }
    if (type2 == null) {
      return type1;
    }
    if (type1 == PythonType.UNKNOWN || type2 == PythonType.UNKNOWN) {
      return PythonType.UNKNOWN;
    }
    if (type1.equals(type2)) {
      return type1;
    }
    Set<PythonType> types = new HashSet<>();
    addTypes(type1, types);
    addTypes(type2, types);
    if (types.size() == 1) {
      return types.iterator().next();
    }
    ensureCandidatesAreNotLazyTypes(types);
    return new UnionType(types);
  }

  private static void addTypes(PythonType type, Set<PythonType> types) {
    if (type instanceof UnionType unionType) {
      types.addAll(unionType.candidates());
    } else {
      types.add(type);
    }
  }

  private static void ensureCandidatesAreNotLazyTypes(Collection<PythonType> types) {
    if (types.stream().anyMatch(LazyType.class::isInstance)) {
      throw new IllegalArgumentException("UnionType cannot contain Lazy types");
    }
  }
}
