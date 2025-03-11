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
package org.sonar.python.types.v2;

import com.google.common.annotations.VisibleForTesting;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;

public class LazyUnionType implements PythonType, ResolvableType {

  private final Set<PythonType> candidates;

  public LazyUnionType(Set<PythonType> candidates) {
    this.candidates = candidates.stream().flatMap(LazyUnionType::flattenLazyUnionTypes).collect(Collectors.toCollection(HashSet::new));
  }

  public PythonType resolve() {
    Set<PythonType> resolvedCandidates = new HashSet<>();
    for (PythonType candidate : candidates) {
      if (candidate instanceof LazyType lazyType) {
        candidate = lazyType.resolve();
      }
      resolvedCandidates.add(candidate);
    }
    return UnionType.or(resolvedCandidates);
  }

  private static Stream<PythonType> flattenLazyUnionTypes(PythonType type) {
    if (type instanceof LazyUnionType lazyUnionType) {
      return lazyUnionType.candidates.stream();
    }
    return Stream.of(type);
  }

  @VisibleForTesting
  protected Set<PythonType> candidates() {
    return Collections.unmodifiableSet(candidates);
  }
}
