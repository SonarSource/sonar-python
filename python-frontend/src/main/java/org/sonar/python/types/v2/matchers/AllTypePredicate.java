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

import java.util.Arrays;
import java.util.List;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.PythonType;

/**
 * TypePredicate implementation that requires all provided type predicates to pass (AND logic).
 * Combines results using TriBool.and() semantics.
 */
class AllTypePredicate implements TypePredicate {

  private final List<TypePredicate> predicates;

  AllTypePredicate(TypePredicate... predicates) {
    this.predicates = Arrays.asList(predicates);
  }

  @Override
  public TriBool check(PythonType type, SubscriptionContext ctx) {
    TriBool result = TriBool.TRUE;
    for (TypePredicate predicate : predicates) {
      TriBool partialResult = predicate.check(type, ctx);
      result = result.logicalAnd(partialResult);
      if (result == TriBool.FALSE) {
        return TriBool.FALSE;
      }
    }
    return result;
  }
}
