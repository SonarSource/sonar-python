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
package org.sonar.python.semantic.v2.types;

import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;

public class LoopAssignment extends Assignment {
  public LoopAssignment(SymbolV2 lhsSymbol, Name lhsName, Expression rhs) {
    super(lhsSymbol, lhsName, rhs);
  }

  @Override
  public PythonType rhsType() {
    PythonType rhsType = super.rhsType();
    if (rhsType instanceof ObjectType objectType && objectType.hasMember("__iter__").isTrue()) {
      // depending on the origin (e.g. typeshed/user-defined), the attribute type may be wrapped in an ObjectType or not,
      // independent of if the type is actually an instance or a class. As such, the logic in the map below assumes
      // the rhsType contains instances of the attribute type, and forces the loop variable type to be an ObjectType.
      var loopVarType = objectType.attributes().stream()
        .findFirst()
        .orElse(PythonType.UNKNOWN);

      if (shouldBeWrappedInObjectType(loopVarType)) {
        return ObjectType.fromType(loopVarType);
      }

      return loopVarType;
    }
    return PythonType.UNKNOWN;
  }

  private static boolean shouldBeWrappedInObjectType(PythonType type) {
    return type instanceof ClassType classType && !"typing.Callable".equals(classType.fullyQualifiedName());
  }
}
