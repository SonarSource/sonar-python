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

import java.util.Set;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

import static org.sonar.python.types.v2.TypeUtils.collectTypes;

public class IsSubtypeOfPredicate implements TypePredicate {
  String fullyQualifiedName;

  public IsSubtypeOfPredicate(String fullyQualifiedName) {
    this.fullyQualifiedName = fullyQualifiedName;
  }

  @Override
  public TriBool check(PythonType type, TypePredicateContext ctx) {
    PythonType expectedType = ctx.typeTable().getType(fullyQualifiedName);

    if (type instanceof UnknownType || expectedType instanceof UnknownType) {
      return TriBool.UNKNOWN;
    }

    Set<PythonType> types = collectTypes(type);
    if (types.stream().anyMatch(t -> t.equals(expectedType))) {
      return TriBool.TRUE;
    } else {
      return TriBool.FALSE;
    }
  }
}
