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
package org.sonar.python.types.v2.matchers;

import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;


public final class InternalTypeMatchers {

  private InternalTypeMatchers() {
  }

  public static TypeMatcher isAnyTypeInUnionSatisfying(TypeMatcher matcher) {
    TypePredicate predicate = getTypePredicate(matcher);
    return new TypeMatcherImpl(new IsAnyTypeInUnionSatisfying(predicate));
  }

  /**
   * Checks whether class decorators permit operand and item checks, unwrapping instances to their classes.
   * Returns TRUE when all decorators in the hierarchy are recognized as harmless for these checks,
   * or class decorators do not apply. Returns FALSE for decorators with unaccounted-for effects,
   * and UNKNOWN for unresolved types or bases. A known unaccounted-for decorator in the hierarchy
   * takes precedence over unresolved bases. Expression unions use the usual conservative matcher semantics.
   * This does not establish member absence, inheritance validity or the effects of metaclasses.
   */
  public static TypeMatcher hasKnownClassDecoratorEffectsInHierarchy() {
    return new TypeMatcherImpl(new HasKnownClassDecoratorEffectsInHierarchyPredicate());
  }

  /**
   * Matches library function signatures with STUB origin and no source definition.
   * Unknown inputs return UNKNOWN; all alternatives of a union must match.
   */
  public static TypeMatcher isStubFunction() {
    return new TypeMatcherImpl(new IsStubFunctionPredicate());
  }

  /**
   * Matches functions with a nonempty decorator list whose entries all satisfy the matcher.
   * Combines decorator matcher results with logical AND: FALSE takes precedence over UNKNOWN.
   * All alternatives of a union must match. No restriction is placed on the function's origin.
   */
  public static TypeMatcher hasOnlyFunctionDecoratorsSatisfying(TypeMatcher matcher) {
    return new TypeMatcherImpl(new HasOnlyFunctionDecoratorsSatisfyingPredicate(getTypePredicate(matcher)));
  }

  /**
   * Matches when an undecorated synchronous function can return a type satisfying the matcher.
   * Callable and return unions match if any alternative matches, and fail only if all fail.
   * Other callable shapes, decorators, asynchronous functions and unresolved types return UNKNOWN.
   */
  public static TypeMatcher canReturnTypeSatisfying(TypeMatcher matcher) {
    return new TypeMatcherImpl(new CanReturnTypeSatisfyingPredicate(getTypePredicate(matcher)));
  }

  private static TypePredicate getTypePredicate(TypeMatcher matcher) {
    if (matcher instanceof TypeMatcherImpl(TypePredicate predicate)) {
      return predicate;
    }
    throw new IllegalArgumentException("Unsupported type matcher: " + matcher.getClass().getName());
  }
}
