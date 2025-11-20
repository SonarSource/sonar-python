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

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.FunctionType;

import static org.assertj.core.api.Assertions.assertThat;

class AnyTypePredicateTest {

  @Test
  void testPredicate() {
    FunctionType function1 = Mockito.mock(FunctionType.class);
    Expression expression1 = Mockito.mock();
    Mockito.when(expression1.typeV2()).thenReturn(function1);


    TypePredicate truePredicate = Mockito.mock(TypePredicate.class);
    Mockito.when(truePredicate.check(Mockito.any(), Mockito.any())).thenReturn(TriBool.TRUE);

    TypePredicate falsePredicate = Mockito.mock(TypePredicate.class);
    Mockito.when(falsePredicate.check(Mockito.any(), Mockito.any())).thenReturn(TriBool.FALSE);

    TypePredicate unknownPredicate = Mockito.mock(TypePredicate.class);
    Mockito.when(unknownPredicate.check(Mockito.any(), Mockito.any())).thenReturn(TriBool.UNKNOWN);

    TypeMatcher trueMatcher = new TypeMatcher(truePredicate);
    TypeMatcher falseMatcher = new TypeMatcher(falsePredicate);
    TypeMatcher unknownMatcher = new TypeMatcher(unknownPredicate);

    TypeMatcher singleTrue = TypeMatchers.any(trueMatcher);
    assertThat(singleTrue.evaluateFor(expression1, Mockito.mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher bothTrue = TypeMatchers.any(trueMatcher, trueMatcher);
    assertThat(bothTrue.evaluateFor(expression1, Mockito.mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher falseTrue = TypeMatchers.any(falseMatcher, trueMatcher);
    assertThat(falseTrue.evaluateFor(expression1, Mockito.mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher trueFalse = TypeMatchers.any(trueMatcher, falseMatcher);
    assertThat(trueFalse.evaluateFor(expression1, Mockito.mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher trueUnknown = TypeMatchers.any(trueMatcher, unknownMatcher);
    assertThat(trueUnknown.evaluateFor(expression1, Mockito.mock())).isEqualTo(TriBool.TRUE);

    TypeMatcher bothUnknown = TypeMatchers.any(unknownMatcher, unknownMatcher);
    assertThat(bothUnknown.evaluateFor(expression1, Mockito.mock())).isEqualTo(TriBool.UNKNOWN);

    TypeMatcher falseFalse = TypeMatchers.any(falseMatcher, falseMatcher);
    assertThat(falseFalse.evaluateFor(expression1, Mockito.mock())).isEqualTo(TriBool.FALSE);

    TypeMatcher falseUnknown = TypeMatchers.any(falseMatcher, unknownMatcher);
    assertThat(falseUnknown.evaluateFor(expression1, Mockito.mock())).isEqualTo(TriBool.UNKNOWN);

    TypeMatcher unknownFalse = TypeMatchers.any(falseMatcher, unknownMatcher);
    assertThat(unknownFalse.evaluateFor(expression1, Mockito.mock())).isEqualTo(TriBool.UNKNOWN);
  }
}
