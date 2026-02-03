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

import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;


public final class InternalTypeMatchers {

  private InternalTypeMatchers() {
  }

  public static TypeMatcher isAnyTypeInUnionSatisfying(TypeMatcher matcher) {
    TypePredicate predicate = getTypePredicate(matcher);
    return new TypeMatcherImpl(new IsAnyTypeInUnionSatisfying(predicate));
  }

  private static TypePredicate getTypePredicate(TypeMatcher matcher) {
    if (matcher instanceof TypeMatcherImpl typeMatcherImpl) {
      return typeMatcherImpl.predicate();
    }
    throw new IllegalArgumentException("Unsupported type matcher: " + matcher.getClass().getName());
  }
}
