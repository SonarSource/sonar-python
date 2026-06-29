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

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.matchers.MatchersTestUtils;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AnyTypePredicateTest {

  @Test
  void testPredicate() {
    FunctionType function1 = mock(FunctionType.class);
    Expression expression1 = mock();
    when(expression1.typeV2()).thenReturn(function1);


    TypePredicate truePredicate = mock(TypePredicate.class);
    when(truePredicate.check(any(), any())).thenReturn(TriBool.TRUE);

    TypePredicate falsePredicate = mock(TypePredicate.class);
    when(falsePredicate.check(any(), any())).thenReturn(TriBool.FALSE);

    TypePredicate unknownPredicate = mock(TypePredicate.class);
    when(unknownPredicate.check(any(), any())).thenReturn(TriBool.UNKNOWN);

    TypeMatcher trueMatcher = MatchersTestUtils.createTypeMatcher(truePredicate);
    TypeMatcher falseMatcher = MatchersTestUtils.createTypeMatcher(falsePredicate);
    TypeMatcher unknownMatcher = MatchersTestUtils.createTypeMatcher(unknownPredicate);

    TypeMatcher singleTrue = TypeMatchers.any(trueMatcher);
    assertThat(singleTrue.evaluateFor(expression1, mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher bothTrue = TypeMatchers.any(trueMatcher, trueMatcher);
    assertThat(bothTrue.evaluateFor(expression1, mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher falseTrue = TypeMatchers.any(falseMatcher, trueMatcher);
    assertThat(falseTrue.evaluateFor(expression1, mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher trueFalse = TypeMatchers.any(trueMatcher, falseMatcher);
    assertThat(trueFalse.evaluateFor(expression1, mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher trueUnknown = TypeMatchers.any(trueMatcher, unknownMatcher);
    assertThat(trueUnknown.evaluateFor(expression1, mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher bothUnknown = TypeMatchers.any(unknownMatcher, unknownMatcher);
    assertThat(bothUnknown.evaluateFor(expression1, mock())).isEqualTo(TriBool.UNKNOWN);

    TypeMatcher falseFalse = TypeMatchers.any(falseMatcher, falseMatcher);
    assertThat(falseFalse.evaluateFor(expression1, mock())).isEqualTo(TriBool.FALSE);

    TypeMatcher falseUnknown = TypeMatchers.any(falseMatcher, unknownMatcher);
    assertThat(falseUnknown.evaluateFor(expression1, mock())).isEqualTo(TriBool.UNKNOWN);

    TypeMatcher unknownFalse = TypeMatchers.any(falseMatcher, unknownMatcher);
    assertThat(unknownFalse.evaluateFor(expression1, mock())).isEqualTo(TriBool.UNKNOWN);
  }
}
