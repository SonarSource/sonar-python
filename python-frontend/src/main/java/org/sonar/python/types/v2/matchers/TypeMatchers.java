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
import java.util.stream.Stream;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.types.v2.TypeSource;

@Beta
public final class TypeMatchers {

  private TypeMatchers() {
  }

  public static TypeMatcher all(Stream<TypeMatcher> matchers) {
    List<TypePredicate> predicates = matchers.map(TypeMatcher::predicate)
      .toList();
    return new TypeMatcher(new AllTypePredicate(predicates));
  }

  public static TypeMatcher all(List<TypeMatcher> matchers) {
    List<TypePredicate> predicates = matchers.stream()
      .map(TypeMatcher::predicate)
      .toList();
    return new TypeMatcher(new AllTypePredicate(predicates));
  }

  public static TypeMatcher all(TypeMatcher... matchers) {
    List<TypePredicate> predicates = Arrays.stream(matchers)
      .map(TypeMatcher::predicate)
      .toList();
    return new TypeMatcher(new AllTypePredicate(predicates));
  }

  public static TypeMatcher any(Stream<TypeMatcher> matchers) {
    List<TypePredicate> predicates = matchers.map(TypeMatcher::predicate)
      .toList();
    return new TypeMatcher(new AnyTypePredicate(predicates));
  }

  public static TypeMatcher any(List<TypeMatcher> matchers) {
    List<TypePredicate> predicates = matchers.stream()
      .map(TypeMatcher::predicate)
      .toList();
    return new TypeMatcher(new AnyTypePredicate(predicates));
  }

  public static TypeMatcher any(TypeMatcher... matchers) {
    List<TypePredicate> predicates = Arrays.stream(matchers)
      .map(TypeMatcher::predicate).toList();
    return new TypeMatcher(new AnyTypePredicate(predicates));
  }

  public static TypeMatcher withFQN(String fqn) {
    return new TypeMatcher(new HasFQNPredicate(fqn));
  }

  public static TypeMatcher isObjectSatisfying(TypeMatcher matcher) {
    TypePredicate predicate = matcher.predicate();
    return new TypeMatcher(new IsObjectSatisfyingPredicate(predicate));
  }

  public static TypeMatcher isType(String fqn) {
    return new TypeMatcher(new IsTypePredicate(fqn));
  }

  public static TypeMatcher isObjectOfType(String fqn) {
    return isObjectSatisfying(isType(fqn));
  }

  public static TypeMatcher isObjectOfSubType(String fqn) {
    return new TypeMatcher(new IsObjectSubtypeOfPredicate(fqn));
  }

  public static TypeMatcher hasTypeSource(TypeSource typeSource) {
    return new TypeMatcher(new TypeSourcePredicate(typeSource));
  }

  public static TypeMatcher hasMember(String memberName) {
    return new TypeMatcher(new HasMemberPredicate(memberName));
  }

  public static TypeMatcher hasMemberSatisfying(String memberName, TypeMatcher matcher) {
    return new TypeMatcher(new HasMemberSatisfyingPredicate(memberName, matcher.predicate()));
  }
}
