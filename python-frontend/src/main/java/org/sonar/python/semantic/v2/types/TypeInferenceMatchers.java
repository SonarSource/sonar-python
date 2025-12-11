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

import java.util.Arrays;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.types.v2.matchers.AnyTypePredicate;
import org.sonar.python.types.v2.matchers.HasFQNPredicate;
import org.sonar.python.types.v2.matchers.IsObjectSatisfyingPredicate;
import org.sonar.python.types.v2.matchers.IsSelfTypePredicate;
import org.sonar.python.types.v2.matchers.IsTypePredicate;
import org.sonar.python.types.v2.matchers.TypePredicate;

/**
 * This class is the entry point for the TypeMatcher API specifically for type inference.
 * Compared to the {@link TypeMatchers} class, this API can handle {@link PythonType} instances directly. 
 * 
 * In case of missing matchers, they should be added to this class, instead of using the constructor directly.
 * 
 * @see TypeInferenceMatcher
 */
class TypeInferenceMatchers {
  private TypeInferenceMatchers() {
  }

  public static TypePredicate any(TypePredicate... predicates) {
    return new AnyTypePredicate(Arrays.asList(predicates));
  }

  public static TypePredicate isType(String fqn) {
    return new IsTypePredicate(fqn);
  }

  public static TypePredicate isObjectSatisfying(TypePredicate matcher) {
    return new IsObjectSatisfyingPredicate(matcher);
  }

  public static TypePredicate isObjectOfType(String fqn) {
    return isObjectSatisfying(isType(fqn));
  }

  public static TypePredicate isSelf() {
    return new IsSelfTypePredicate();
  }

  public static TypePredicate withFQN(String fqn) {
    return new HasFQNPredicate(fqn);
  }
}
