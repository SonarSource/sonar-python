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
package org.sonar.python.types.v2.matchers;

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Set;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnknownType;

/**
 * TypePredicate decorator that checks if a type or any of its supertypes satisfies the wrapped predicate.
 * Traverses the class hierarchy using BFS to check the type itself and all its superclasses.
 * Note: UnionType handling is performed by TypeMatcherImpl, not by this predicate.
 */
public class IsTypeOrSuperTypeSatisfyingPredicate implements TypePredicate {

  private final TypePredicate wrappedPredicate;

  public IsTypeOrSuperTypeSatisfyingPredicate(TypePredicate wrappedPredicate) {
    this.wrappedPredicate = wrappedPredicate;
  }

  @Override
  public TriBool check(PythonType type, TypePredicateContext ctx) {
    if (type instanceof UnknownType) {
      return TriBool.UNKNOWN;
    }

    Set<PythonType> visited = new HashSet<>();
    ArrayDeque<PythonType> queue = new ArrayDeque<>();
    queue.add(type);

    boolean hasUnknown = false;

    while (!queue.isEmpty()) {
      PythonType currentType = queue.poll();

      if (visited.contains(currentType)) {
        continue;
      }
      visited.add(currentType);

      TriBool result = wrappedPredicate.check(currentType, ctx);
      if (result == TriBool.TRUE) {
        return TriBool.TRUE;
      } else if (result == TriBool.UNKNOWN) {
        hasUnknown = true;
      }

      if (currentType instanceof ClassType classType) {
        for (TypeWrapper superClass : classType.superClasses()) {
          queue.add(superClass.type());
        }
      }
    }

    return hasUnknown ? TriBool.UNKNOWN : TriBool.FALSE;
  }
}
