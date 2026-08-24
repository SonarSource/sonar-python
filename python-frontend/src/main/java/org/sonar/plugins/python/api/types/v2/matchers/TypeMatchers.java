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
package org.sonar.plugins.python.api.types.v2.matchers;

import com.google.common.annotations.VisibleForTesting;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.python.types.v2.matchers.AllTypePredicate;
import org.sonar.python.types.v2.matchers.AnyTypePredicate;
import org.sonar.python.types.v2.matchers.HasFQNPredicate;
import org.sonar.python.types.v2.matchers.HasFQNSatisfyingPredicate;
import org.sonar.python.types.v2.matchers.HasMemberPredicate;
import org.sonar.python.types.v2.matchers.HasMemberSatisfyingPredicate;
import org.sonar.python.types.v2.matchers.IsFunctionOwnerSatisfyingPredicate;
import org.sonar.python.types.v2.matchers.IsObjectSatisfyingPredicate;
import org.sonar.python.types.v2.matchers.IsTypeOrSuperTypeSatisfyingPredicate;
import org.sonar.python.types.v2.matchers.IsTypePredicate;
import org.sonar.python.types.v2.matchers.TypeMatcherImpl;
import org.sonar.python.types.v2.matchers.TypePredicate;
import org.sonar.python.types.v2.matchers.TypeSourcePredicate;

public final class TypeMatchers {

  private TypeMatchers() {
  }

  public static TypeMatcher all(Stream<TypeMatcher> matchers) {
    List<TypePredicate> predicates = matchers.map(TypeMatchers::getTypePredicate)
      .toList();
    return new TypeMatcherImpl(new AllTypePredicate(predicates));
  }

  public static TypeMatcher all(List<TypeMatcher> matchers) {
    List<TypePredicate> predicates = matchers.stream()
      .map(TypeMatchers::getTypePredicate)
      .toList();
    return new TypeMatcherImpl(new AllTypePredicate(predicates));
  }

  public static TypeMatcher all(TypeMatcher... matchers) {
    List<TypePredicate> predicates = Arrays.stream(matchers)
      .map(TypeMatchers::getTypePredicate)
      .toList();
    return new TypeMatcherImpl(new AllTypePredicate(predicates));
  }

  public static TypeMatcher any(Stream<TypeMatcher> matchers) {
    List<TypePredicate> predicates = matchers.map(TypeMatchers::getTypePredicate)
      .toList();
    return new TypeMatcherImpl(new AnyTypePredicate(predicates));
  }

  public static TypeMatcher any(List<TypeMatcher> matchers) {
    List<TypePredicate> predicates = matchers.stream()
      .map(TypeMatchers::getTypePredicate)
      .toList();
    return new TypeMatcherImpl(new AnyTypePredicate(predicates));
  }

  public static TypeMatcher any(TypeMatcher... matchers) {
    List<TypePredicate> predicates = Arrays.stream(matchers)
      .map(TypeMatchers::getTypePredicate).toList();
    return new TypeMatcherImpl(new AnyTypePredicate(predicates));
  }

  public static TypeMatcher withFQN(String fqn) {
    return new TypeMatcherImpl(new HasFQNPredicate(fqn));
  }

  /**
   * Matches types whose FQN satisfies {@code fqnPredicate}. Prefer {@link #isType(String)} when
   * stubs exist; FQN-based matching is a fallback for unresolved third-party imports.
   */
  public static TypeMatcher withFQNSatisfying(Function<String, TriBool> fqnPredicate) {
    return new TypeMatcherImpl(new HasFQNSatisfyingPredicate(fqnPredicate));
  }

  /**
   * Matches types whose FQN starts with {@code fqnPrefix}. Prefer a trailing {@code '.'}
   * (e.g. {@code "numpy."}) so sibling modules like {@code numpyish} are not matched.
   * Prefer {@link #isType(String)} when stubs exist; FQN-based matching is a fallback for
   * unresolved third-party imports.
   */
  public static TypeMatcher withFQNPrefix(String fqnPrefix) {
    return withFQNSatisfying(typeFqn -> typeFqn.startsWith(fqnPrefix) ? TriBool.TRUE : TriBool.FALSE);
  }

  public static TypeMatcher isObjectSatisfying(TypeMatcher matcher) {
    TypePredicate predicate = getTypePredicate(matcher);
    return new TypeMatcherImpl(new IsObjectSatisfyingPredicate(predicate));
  }

  public static TypeMatcher isType(String fqn) {
    return new TypeMatcherImpl(new IsTypePredicate(fqn));
  }

  public static TypeMatcher isObjectOfType(String fqn) {
    return isObjectSatisfying(isType(fqn));
  }

  /**
   * Checks if the type of the expression is an object type and if its nested type, or its supertypes, matches the given type by equality.
   * @param fqn The FQN of the type to match by equality
   * @return a type matcher
   */
  public static TypeMatcher isObjectInstanceOf(String fqn) {
    return isObjectSatisfying(isOrExtendsType(fqn));
  }

  public static TypeMatcher isOrExtendsType(String fqn) {
    return isTypeOrSuperTypeSatisfying(isType(fqn));
  }

  public static TypeMatcher isTypeOrSuperTypeWithFQN(String fqn) {
    return isTypeOrSuperTypeSatisfying(withFQN(fqn));
  }

  public static TypeMatcher isTypeOrSuperTypeSatisfying(TypeMatcher matcher) {
    TypePredicate predicate = getTypePredicate(matcher);
    return new TypeMatcherImpl(new IsTypeOrSuperTypeSatisfyingPredicate(predicate));
  }

  public static TypeMatcher isFunctionOwnerSatisfying(TypeMatcher matcher) {
    TypePredicate predicate = getTypePredicate(matcher);
    return new TypeMatcherImpl(new IsFunctionOwnerSatisfyingPredicate(predicate));
  }

  public static TypeMatcher hasTypeSource(TypeSource typeSource) {
    return new TypeMatcherImpl(new TypeSourcePredicate(typeSource));
  }

  public static TypeMatcher hasMember(String memberName) {
    return new TypeMatcherImpl(new HasMemberPredicate(memberName));
  }

  public static TypeMatcher hasMemberSatisfying(String memberName, TypeMatcher matcher) {
    return new TypeMatcherImpl(new HasMemberSatisfyingPredicate(memberName, getTypePredicate(matcher)));
  }

  @VisibleForTesting
  static TypePredicate getTypePredicate(TypeMatcher matcher) {
    if (matcher instanceof TypeMatcherImpl(TypePredicate predicate)) {
      return predicate;
    }
    throw new IllegalArgumentException("Unsupported type matcher: " + matcher.getClass().getName());
  }
}
