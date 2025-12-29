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
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.plugins.python.api.types.v2.matchers.MatchersTestUtils;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.v2.TestProject;

import static org.assertj.core.api.Assertions.assertThat;

class HasMemberPredicateTest {

  @Test
  void testHasMember() {
    var project = new TestProject();
    project.addModule("my_file.py", """
    class A:
      def foo(self):
        pass
    """);
    Expression objectTypeExpression = project.lastExpression("""
      from my_file import A
      a = A()
      a
      """);

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());
    TypePredicateContext predicateContext = TypePredicateContext.of(project.projectLevelTypeTable());

    HasMemberPredicate hasFooTypePredicate = new HasMemberPredicate("foo");
    HasMemberPredicate hasBarTypePredicate = new HasMemberPredicate("bar");

    PythonType objectType = objectTypeExpression.typeV2();

    assertThat(hasFooTypePredicate.check(objectType, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(hasBarTypePredicate.check(objectType, predicateContext)).isEqualTo(TriBool.FALSE);

    assertThat(TypeMatchers.hasMember("foo").isTrueFor(objectTypeExpression, ctx)).isTrue();
    assertThat(TypeMatchers.hasMember("bar").isTrueFor(objectTypeExpression, ctx)).isFalse();

    assertThat(objectType).isNotInstanceOf(UnknownType.class);

    TypePredicate truePredicate = Mockito.mock(TypePredicate.class);
    TypePredicate falsePredicate = Mockito.mock(TypePredicate.class);
    TypePredicate unknownPredicate = Mockito.mock(TypePredicate.class);
    Mockito.when(truePredicate.check(Mockito.any(), Mockito.any())).thenReturn(TriBool.TRUE);
    Mockito.when(falsePredicate.check(Mockito.any(), Mockito.any())).thenReturn(TriBool.FALSE);
    Mockito.when(unknownPredicate.check(Mockito.any(), Mockito.any())).thenReturn(TriBool.UNKNOWN);

    TypeMatcher trueMatcher = MatchersTestUtils.createTypeMatcher(truePredicate);
    TypeMatcher falseMatcher = MatchersTestUtils.createTypeMatcher(falsePredicate);
    TypeMatcher unknownMatcher = MatchersTestUtils.createTypeMatcher(unknownPredicate);

    assertThat(TypeMatchers.hasMemberSatisfying("foo", trueMatcher).isTrueFor(objectTypeExpression, ctx)).isTrue();
    assertThat(TypeMatchers.hasMemberSatisfying("foo", falseMatcher).isTrueFor(objectTypeExpression, ctx)).isFalse();
    assertThat(TypeMatchers.hasMemberSatisfying("foo", unknownMatcher).evaluateFor(objectTypeExpression, ctx).isUnknown()).isTrue();
    assertThat(TypeMatchers.hasMemberSatisfying("bar", trueMatcher).evaluateFor(objectTypeExpression, ctx).isFalse()).isTrue();
  }
}
