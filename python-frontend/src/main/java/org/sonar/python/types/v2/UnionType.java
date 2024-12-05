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
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.api.Beta;

@Beta
public class UnionType implements PythonType {

  private final Set<PythonType> candidates = new HashSet<>();

  private UnionType(Set<PythonType> candidates) {
    this.candidates.addAll(candidates);
  }

  public Set<PythonType> candidates() {
    return candidates;
  }

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

  @Override
  public boolean equals(Object o) {
    if (o == null || getClass() != o.getClass()) return false;
    UnionType unionType = (UnionType) o;
    return Objects.equals(candidates, unionType.candidates);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(candidates);
  }

  @Override
  public String toString() {
    return displayName().orElse(super.toString());
  }

  public static PythonType or(@Nullable PythonType type1, @Nullable PythonType type2, @Nullable PythonType ...types) {
    if(types == null) {
      types = new PythonType[0];
    }
    Set<PythonType> typeSet = new HashSet<>();
    typeSet.add(type1);
    typeSet.add(type2);
    typeSet.addAll(Set.of(types));
    return or(typeSet);
  }

  public static PythonType or(Collection<PythonType> types) {
    types = types.stream().filter(Objects::nonNull).collect(Collectors.toSet());
    if(types.isEmpty()) {
      return PythonType.UNKNOWN;
    }
    if(types.size() == 1) {
      return types.iterator().next();
    }

    Set<PythonType> flatTypes = types.stream().flatMap(UnionType::flattenPythonType).collect(Collectors.toSet());
    if(flatTypes.stream().anyMatch(type -> type == PythonType.UNKNOWN)) {
      return PythonType.UNKNOWN;
    }
    ensureCandidatesAreNotLazyTypes(flatTypes);
    return new UnionType(flatTypes);
  }

  private static Stream<PythonType> flattenPythonType(PythonType type) {
    if(type instanceof UnionType unionType) {
      return unionType.candidates().stream();
    } else {
      return Stream.of(type);
    }
  }

  private static void ensureCandidatesAreNotLazyTypes(Collection<PythonType> types) {
    if (types.stream().anyMatch(LazyType.class::isInstance)) {
      throw new IllegalArgumentException("UnionType cannot contain Lazy types");
    }
  }
}
