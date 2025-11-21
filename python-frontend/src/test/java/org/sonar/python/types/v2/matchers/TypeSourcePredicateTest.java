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
import org.sonar.plugins.python.api.types.v2.TypeSource;

import static org.assertj.core.api.Assertions.assertThat;

class TypeSourcePredicateTest {

  @ParameterizedTest
  @MethodSource("matchingTypeSourceCases")
  void testMatchingTypeSources(TypeSource typeSource) {
    PythonType type = createTypeWithSource(typeSource);
    TypeSourcePredicate predicate = new TypeSourcePredicate(typeSource);

    assertThat(predicate.check(type, null)).isEqualTo(TriBool.TRUE);
  }

  @ParameterizedTest
  @MethodSource("nonMatchingTypeSourceCases")
  void testNonMatchingTypeSources(TypeSource typeTypeSource, TypeSource predicateTypeSource) {
    PythonType type = createTypeWithSource(typeTypeSource);
    TypeSourcePredicate predicate = new TypeSourcePredicate(predicateTypeSource);

    assertThat(predicate.check(type, null)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void testThroughTypeMatchers() {
    PythonType typeWithExactSource = createTypeWithSource(TypeSource.EXACT);
    PythonType typeWithTypeHintSource = createTypeWithSource(TypeSource.TYPE_HINT);
    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);

    assertThat(TypeMatchers.hasTypeSource(TypeSource.EXACT).predicate().check(typeWithExactSource, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.hasTypeSource(TypeSource.EXACT).predicate().check(typeWithTypeHintSource, ctx)).isEqualTo(TriBool.FALSE);
    assertThat(TypeMatchers.hasTypeSource(TypeSource.TYPE_HINT).predicate().check(typeWithTypeHintSource, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.hasTypeSource(TypeSource.TYPE_HINT).predicate().check(typeWithExactSource, ctx)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void testObjectTypeIsNotUnwrapped() {
    PythonType wrappedType = createTypeWithSource(TypeSource.TYPE_HINT);

    ObjectType objectType = Mockito.mock(ObjectType.class);
    Mockito.when(objectType.typeSource()).thenReturn(TypeSource.EXACT);
    Mockito.when(objectType.unwrappedType()).thenReturn(wrappedType);

    TypeSourcePredicate exactPredicate = new TypeSourcePredicate(TypeSource.EXACT);
    TypeSourcePredicate typeHintPredicate = new TypeSourcePredicate(TypeSource.TYPE_HINT);

    assertThat(exactPredicate.check(objectType, null)).isEqualTo(TriBool.TRUE);
    assertThat(typeHintPredicate.check(objectType, null)).isEqualTo(TriBool.FALSE);
  }

  private static Stream<TypeSource> matchingTypeSourceCases() {
    return Stream.of(TypeSource.EXACT, TypeSource.TYPE_HINT);
  }

  private static Stream<Arguments> nonMatchingTypeSourceCases() {
    return Stream.of(
      Arguments.of(TypeSource.TYPE_HINT, TypeSource.EXACT),
      Arguments.of(TypeSource.EXACT, TypeSource.TYPE_HINT)
    );
  }

  private static PythonType createTypeWithSource(TypeSource typeSource) {
    PythonType type = Mockito.mock(PythonType.class);
    Mockito.when(type.typeSource()).thenReturn(typeSource);
    return type;
  }
}

