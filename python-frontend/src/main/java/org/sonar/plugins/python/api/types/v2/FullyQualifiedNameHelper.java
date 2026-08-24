/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
    return switch (type) {
      case SelfType selfType -> getFullyQualifiedName(selfType.innerType());
      case FunctionType functionType -> Optional.ofNullable(functionType.fullyQualifiedName());
      case ClassType classType -> Optional.ofNullable(classType.fullyQualifiedName());
      case ModuleType moduleType -> Optional.ofNullable(moduleType.fullyQualifiedName());
      case SpecialFormType specialFormType -> Optional.ofNullable(specialFormType.fullyQualifiedName());
      case UnknownType.UnresolvedImportType(String importPath) -> Optional.ofNullable(importPath);
      default -> Optional.empty();
    };
  }
}
