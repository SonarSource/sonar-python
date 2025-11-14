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

import java.util.function.UnaryOperator;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

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

  @CheckForNull
  public static String getFullyQualifiedName(PythonType type) {
    if (type instanceof FunctionType functionType) {
      return functionType.fullyQualifiedName();
    } else if (type instanceof ClassType classType) {
      return classType.fullyQualifiedName();
    } else if (type instanceof ModuleType moduleType) {
      return moduleType.fullyQualifiedName();
    } else if (type instanceof SpecialFormType specialFormType) {
      return specialFormType.fullyQualifiedName();
    } else if (type instanceof UnknownType.UnresolvedImportType unresolvedImportType) {
      return unresolvedImportType.importPath();
    }
    return null;
  }
}
