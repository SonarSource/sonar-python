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
package org.sonar.python.types.v2.matchs;

import com.google.common.annotations.VisibleForTesting;
import java.util.Set;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;

@Beta
public class TypeMatcher {

  private final TypePredicate predicate;

  public TypeMatcher(TypePredicate predicate) {
    this.predicate = predicate;
  }

  public TriBool isFor(Expression expr, SubscriptionContext ctx) {
    PythonType type = expr.typeV2();
    Set<PythonType> candidates = extractCandidates(type);

    TriBool result = TriBool.TRUE;
    for (PythonType candidate : candidates) {
      result = result.and(predicate.check(candidate, ctx));
      if (result.isFalse() || result.isUnknown()) {
        break;
      }
    }
    return result;
  }

  public TriBool canBeFor(Expression expr, SubscriptionContext ctx) {
    PythonType type = expr.typeV2();
    Set<PythonType> candidates = extractCandidates(type);
    TriBool result = TriBool.FALSE;
    for (PythonType candidate : candidates) {
      result = result.or(predicate.check(candidate, ctx));
      if (result.isTrue()) {
        break;
      }
    }
    return result;
  }

  public boolean isTrueFor(Expression expr, SubscriptionContext ctx) {
    return isFor(expr, ctx).isTrue();
  }

  public boolean canBeTrueFor(Expression expr, SubscriptionContext ctx) {
    return canBeFor(expr, ctx).isTrue();
  }

  @VisibleForTesting
  static Set<PythonType> extractCandidates(PythonType type) {
    if (type instanceof UnionType unionType) {
      return unionType.candidates();
    }
    return Set.of(type);
  }
}
