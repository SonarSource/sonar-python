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

import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

/**
 * TypePredicate decorator that unwraps FunctionType to check if its owner satisfies the wrapped predicate.
 * Returns FALSE if the type is not a FunctionType or if the owner is null.
 * Similar pattern to IsObjectSatisfyingPredicate but for FunctionType → owner relationship.
 */
public class IsFunctionOwnerSatisfyingPredicate implements TypePredicate {

  private final TypePredicate wrappedPredicate;

  public IsFunctionOwnerSatisfyingPredicate(TypePredicate wrappedPredicate) {
    this.wrappedPredicate = wrappedPredicate;
  }

  @Override
  public TriBool check(PythonType type, TypePredicateContext ctx) {
    if (type instanceof UnknownType) {
      return TriBool.UNKNOWN;
    }

    if (!(type instanceof FunctionType functionType)) {
      return TriBool.FALSE;
    }

    PythonType owner = functionType.owner();
    if (owner == null) {
      return TriBool.FALSE;
    }

    return wrappedPredicate.check(owner, ctx);
  }
}
