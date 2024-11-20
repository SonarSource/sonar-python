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

import java.util.HashSet;
import java.util.Set;

public class LazyUnionType implements PythonType, ResolvableType {

  Set<PythonType> candidates;

  public LazyUnionType(Set<PythonType> candidates) {
    this.candidates = candidates;
  }

  public PythonType resolve() {
    Set<PythonType> resolvedCandidates = new HashSet<>();
    for (PythonType candidate : candidates) {
      if (candidate instanceof LazyType lazyType) {
        candidate = lazyType.resolve();
      }
      resolvedCandidates.add(candidate);
    }
    return new UnionType(resolvedCandidates);
  }
}
