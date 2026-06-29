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

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.plugins.python.api.types.v2.matchers.MatchersTestUtils;
import org.sonar.python.semantic.v2.typetable.TypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class IsAnyTypeInUnionSatisfyingTest {

  @Test
  void testUnionTypeWithOneMatchingCandidate() {
    PythonType candidate1 = mock(PythonType.class);
    PythonType candidate2 = mock(PythonType.class);
    UnionType unionType = (UnionType) UnionType.or(candidate1, candidate2);

    TypePredicate wrappedPredicate = MatchersTestUtils.mockPredicateReturning(candidate1, TriBool.TRUE);
    when(wrappedPredicate.check(eq(candidate2), any())).thenReturn(TriBool.FALSE);

    TriBool result = checkType(unionType, wrappedPredicate);
    assertThat(result).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testUnionTypeWithAllNonMatchingCandidates() {
    PythonType candidate1 = mock(PythonType.class);
    PythonType candidate2 = mock(PythonType.class);
    UnionType unionType = (UnionType) UnionType.or(candidate1, candidate2);

    TypePredicate wrappedPredicate = MatchersTestUtils.mockPredicateReturning(candidate1, TriBool.FALSE);
    when(wrappedPredicate.check(eq(candidate2), any())).thenReturn(TriBool.FALSE);

    TriBool result = checkType(unionType, wrappedPredicate);
    assertThat(result).isEqualTo(TriBool.FALSE);
  }

  @Test
  void testUnionTypeWithUnknownCandidate() {
    PythonType candidate1 = mock(PythonType.class);
    PythonType candidate2 = mock(PythonType.class);
    UnionType unionType = (UnionType) UnionType.or(candidate1, candidate2);

    TypePredicate wrappedPredicate = MatchersTestUtils.mockPredicateReturning(candidate1, TriBool.FALSE);
    when(wrappedPredicate.check(eq(candidate2), any())).thenReturn(TriBool.UNKNOWN);

    TriBool result = checkType(unionType, wrappedPredicate);
    assertThat(result).isEqualTo(TriBool.UNKNOWN);
  }


  @ParameterizedTest
  @MethodSource("nonUnionTypesWithExpectedResults")
  void testNonUnionTypeDoesNotDelegateToWrappedPredicate(PythonType nonUnionType, TriBool expectedResult) {
    TypePredicate wrappedPredicate = mock(TypePredicate.class);

    TriBool result = checkType(nonUnionType, wrappedPredicate);

    assertThat(result).isEqualTo(expectedResult);
    verifyNoInteractions(wrappedPredicate);
  }

  static Stream<Arguments> nonUnionTypesWithExpectedResults() {
    return Stream.of(
      Arguments.of(PythonType.UNKNOWN, TriBool.UNKNOWN),
      Arguments.of(new UnknownType.UnresolvedImportType("some.module"), TriBool.UNKNOWN),
      Arguments.of(mock(PythonType.class), TriBool.FALSE)
    );
  }

  private static TriBool checkType(PythonType type, TypePredicate wrappedPredicate) {
    IsAnyTypeInUnionSatisfying predicate = new IsAnyTypeInUnionSatisfying(wrappedPredicate);
    TypePredicateContext ctx = TypePredicateContext.of(mock(TypeTable.class));
    return predicate.check(type, ctx);
  }
}
