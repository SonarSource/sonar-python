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
package org.sonar.plugins.python.api.types.v2;

import java.util.Optional;
import org.sonar.api.Beta;
import org.sonar.python.types.v2.SpecialFormType;

@Beta
public class FullyQualifiedNameHelper {
  private FullyQualifiedNameHelper() {
  }


  @Beta
  public static Optional<String> getFullyQualifiedName(PythonType type) {
    if (type instanceof SelfType selfType) {
      return getFullyQualifiedName(selfType.innerType());
    } else if (type instanceof FunctionType functionType) {
      return Optional.ofNullable(functionType.fullyQualifiedName());
    } else if (type instanceof ClassType classType) {
      return Optional.ofNullable(classType.fullyQualifiedName());
    } else if (type instanceof ModuleType moduleType) {
      return Optional.ofNullable(moduleType.fullyQualifiedName());
    } else if (type instanceof SpecialFormType specialFormType) {
      return Optional.ofNullable(specialFormType.fullyQualifiedName());
    } else if (type instanceof UnknownType.UnresolvedImportType unresolvedImportType) {
      return Optional.ofNullable(unresolvedImportType.importPath());
    }
    return Optional.empty();
  }
}
