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
package org.sonar.python.semantic.v2.dynamicstubs;

import java.util.Map;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;

public class DynamicStubsProvider {

  private DynamicStubsProvider() {
  }

  public static Map<String, TypeWrapper> createDynamicStubs(String moduleFqn) {
    return switch (moduleFqn) {
      case "typing" -> typingDynamicStubs();
      default -> Map.of();
    };
  }

  /**
   * A lot of the types exported by the typing module need a special representation in the internal type system. 
   * Furthermore, mypy isn't able to analyze moduels called "typing", as it clashes internally with the standard 
   * library's typing module (see https://github.com/python/mypy/issues/1876). As a result, this function allows us 
   * to create the proper special types for the typing module.
   * 
   * @return a map of the stubs for the typing module
   */
  private static Map<String, TypeWrapper> typingDynamicStubs() {
    return Map.of(
      // Every type is compatible with Any, and Any is compatible with every type. This is represented as UNKNOWN in PythonType
      "Any", TypeWrapper.of(PythonType.UNKNOWN));
  }
}
