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
package org.sonar.python.api.types.v2.matchers;

import com.google.common.annotations.VisibleForTesting;
import java.util.Set;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.python.types.v2.matchers.TypePredicate;

// This record is package-private
@Beta
record TypeMatcherImpl(TypePredicate predicate) implements TypeMatcher {

  @Override
  public TriBool evaluateFor(Expression expr, SubscriptionContext ctx) {
    PythonType type = expr.typeV2();
    Set<PythonType> candidates = extractCandidates(type);

    TriBool result = TriBool.TRUE;
    for (PythonType candidate : candidates) {
      result = result.conservativeAnd(predicate.check(candidate, ctx));
      if (result.isUnknown()) {
        break;
      }
    }
    return result;
  }

  @Override
  public boolean isTrueFor(Expression expr, SubscriptionContext ctx) {
    return evaluateFor(expr, ctx).isTrue();
  }

  @VisibleForTesting
  static Set<PythonType> extractCandidates(PythonType type) {
    if (type instanceof UnionType unionType) {
      return unionType.candidates();
    }
    return Set.of(type);
  }
}
