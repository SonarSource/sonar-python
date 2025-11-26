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
package org.sonar.python.types.v2;

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Set;
import java.util.function.UnaryOperator;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;

public class TypeUtils {

  private TypeUtils() {
  }

  public static PythonType resolved(PythonType pythonType) {
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

  public static PythonType map(PythonType type, UnaryOperator<PythonType> mapper) {
    if (type instanceof UnionType unionType) {
      return unionType.candidates().stream().map(mapper).collect(toUnionType());
    } else {
      return mapper.apply(type);
    }
  }

  public static Collector<PythonType, ?, PythonType> toUnionType() {
    return Collectors.collectingAndThen(Collectors.toSet(), UnionType::or);
  }

  public static Set<PythonType> collectTypes(PythonType type) {
    var result = new HashSet<PythonType>();
    var queue = new ArrayDeque<PythonType>();
    queue.add(type);
    while (!queue.isEmpty()) {
      var currentType = queue.pop();
      if (result.contains(currentType)) {
        continue;
      }
      result.add(currentType);
      if (currentType instanceof UnionType) {
        result.clear();
        result.add(PythonType.UNKNOWN);
        queue.clear();
      } else if (currentType instanceof ClassType classType) {
        queue.addAll(classType.superClasses().stream().map(TypeWrapper::type).toList());
      }
    }
    return result;
  }
}
