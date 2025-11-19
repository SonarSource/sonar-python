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

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

import static org.assertj.core.api.Assertions.assertThat;

class IsObjectSatisfyingPredicateTest {

  @ParameterizedTest
  @MethodSource("objectTypeWithWrappedPredicateResults")
  void testObjectTypeUnwrapsAndDelegatesToWrappedPredicate(TriBool wrappedPredicateResult, TriBool expectedResult) {
    PythonType wrappedType = Mockito.mock(PythonType.class);
    ObjectType objectType = new ObjectType(wrappedType);
    TypePredicate wrappedPredicate = MatchersTestUtils.mockPredicateReturning(wrappedType, wrappedPredicateResult);

    TriBool result = checkType(objectType, wrappedPredicate);
    assertThat(result).isEqualTo(expectedResult);
  }

  static Stream<Arguments> objectTypeWithWrappedPredicateResults() {
    return Stream.of(
      Arguments.of(TriBool.TRUE, TriBool.TRUE),
      Arguments.of(TriBool.FALSE, TriBool.FALSE),
      Arguments.of(TriBool.UNKNOWN, TriBool.UNKNOWN)
    );
  }

  @ParameterizedTest
  @MethodSource("nonObjectTypesWithExpectedResults")
  void testNonObjectTypeDoesNotDelegateToWrappedPredicate(PythonType nonObjectType, TriBool expectedResult) {
    TypePredicate wrappedPredicate = Mockito.mock(TypePredicate.class);

    TriBool result = checkType(nonObjectType, wrappedPredicate);

    assertThat(result).isEqualTo(expectedResult);
    Mockito.verifyNoInteractions(wrappedPredicate);
  }


  static Stream<Arguments> nonObjectTypesWithExpectedResults() {
    return Stream.of(
      Arguments.of(PythonType.UNKNOWN, TriBool.UNKNOWN),
      Arguments.of(new UnknownType.UnresolvedImportType("some.module"), TriBool.UNKNOWN),
      Arguments.of(Mockito.mock(PythonType.class), TriBool.FALSE)
    );
  }

  @Test
  void testTypeMatchersIsObjectSatisfyingIntegration() {
    PythonType wrappedType = Mockito.mock(PythonType.class);
    ObjectType objectType = new ObjectType(wrappedType);

    TypePredicate innerPredicate = MatchersTestUtils.mockPredicateReturning(wrappedType, TriBool.TRUE);
    TypeMatcher innerMatcher = new TypeMatcher(innerPredicate);

    TypeMatcher objectThatMatcher = TypeMatchers.isObjectSatisfying(innerMatcher);

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    TriBool result = objectThatMatcher.predicate().check(objectType, ctx);

    assertThat(result).isEqualTo(TriBool.TRUE);
  }

  private static TriBool checkType(PythonType type, TypePredicate wrappedPredicate) {
    IsObjectSatisfyingPredicate predicate = new IsObjectSatisfyingPredicate(wrappedPredicate);
    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    return predicate.check(type, ctx);
  }
}

