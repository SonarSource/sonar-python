/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.types.v2;

import java.util.Set;
import java.util.stream.Collectors;

public class TypeUtils {

  private TypeUtils() {

  }

  static PythonType resolved(PythonType pythonType) {
    if (pythonType instanceof ResolvableType resolvableType) {
      return resolvableType.resolve();
    }
    return pythonType;
  }

  public static PythonType ensureWrappedObjectType(PythonType pythonType) {
    if (!(pythonType instanceof ObjectType)) {
      return new ObjectType(pythonType);
    }
    return pythonType;
  }


  /**
   * @param type the input type
   * @return a set of all possible effective types. However, {@link ObjectType}s are not unwrapped, as they are used to represent an
   * instance of a type, rather than a type itself, and this should be done explicitly.
   */
  public static Set<PythonType> getNestedEffectiveTypes(PythonType type) {
    if (type instanceof ResolvableType resolvableType) {
      type = resolvableType.resolve();
    }
    if (type instanceof UnionType unionType) {
      return unionType.candidates().stream()
        .map(TypeUtils::getNestedEffectiveTypes)
        .flatMap(Set::stream)
        .collect(Collectors.toSet());
    }
    return Set.of(type);
  }

  public static PythonType unwrapType(PythonType type) {
    if (type instanceof ObjectType objectType) {
      return unwrapType(objectType.unwrappedType());
    }
    if (type instanceof UnionType unionType) {
      var newCandidates = unionType.candidates().stream()
        .map(TypeUtils::unwrapType)
        .collect(Collectors.toSet());
      return UnionType.or(newCandidates);
    }
    return type;
  }
}
