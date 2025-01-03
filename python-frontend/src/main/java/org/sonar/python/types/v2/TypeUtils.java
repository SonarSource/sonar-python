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

import java.util.function.UnaryOperator;
import java.util.stream.Collector;
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
}
