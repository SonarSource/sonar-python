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
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.v2.TestProject;

import static org.assertj.core.api.Assertions.assertThat;

class IsSubtypeOfPredicateTest {

  @Test
  void testCheckForObjectType() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A :
        def __init__(self):
          pass
      class B(A):
        def __init__(self):
          pass
      class C(A):
        def __init__(self):
          pass
      """);
    Expression objectTypeExpression = project.lastExpression("""
      from my_file import B
      b = B()
      b
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    SubscriptionContext subscriptionContext = Mockito.mock(SubscriptionContext.class);
    Mockito.when(subscriptionContext.typeTable()).thenReturn(project.projectLevelTypeTable());

    IsSubtypeOfPredicate isSubtypeOfBPredicate = new IsSubtypeOfPredicate("my_file.B");
    IsSubtypeOfPredicate isSubtypeOfAPredicate = new IsSubtypeOfPredicate("my_file.A");
    IsSubtypeOfPredicate isSubtypeOfCPredicate = new IsSubtypeOfPredicate("my_file.C");

    TypeMatcher isObjectOfSubtypeBMatcher = TypeMatchers.isObjectOfSubType("my_file.B");
    TypeMatcher isObjectOfSubtypeAMatcher = TypeMatchers.isObjectOfSubType("my_file.A");
    TypeMatcher isObjectOfSubtypeCMatcher = TypeMatchers.isObjectOfSubType("my_file.C");

    assertThat(isSubtypeOfAPredicate.check(objectTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);
    assertThat(TypeMatchers.isSubtypeOf("my_file.A").evaluateFor(objectTypeExpression, subscriptionContext)).isEqualTo(TriBool.FALSE);
    assertThat(isSubtypeOfBPredicate.check(objectTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);

    assertThat(isSubtypeOfCPredicate.check(objectTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);

    assertThat(isObjectOfSubtypeBMatcher.evaluateFor(objectTypeExpression, subscriptionContext)).isEqualTo(TriBool.TRUE);
    assertThat(isObjectOfSubtypeAMatcher.evaluateFor(objectTypeExpression, subscriptionContext)).isEqualTo(TriBool.TRUE);
    assertThat(isObjectOfSubtypeCMatcher.evaluateFor(objectTypeExpression, subscriptionContext)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void testCheckForClassType() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A :
        def __init__(self):
          pass
      class B(A):
        def __init__(self):
          pass
      """);
    Expression classTypeExpression = project.lastExpression("""
      from my_file import B
      B
      """);

    var ctx = TypePredicateContext.of(project.projectLevelTypeTable());
    IsSubtypeOfPredicate isSubtypeOfAPredicate = new IsSubtypeOfPredicate("my_file.A");
    assertThat(isSubtypeOfAPredicate.check(classTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);

    SubscriptionContext subscriptionContext = Mockito.mock(SubscriptionContext.class);
    Mockito.when(subscriptionContext.typeTable()).thenReturn(project.projectLevelTypeTable());
    assertThat(TypeMatchers.isObjectOfSubType("my_file.A").evaluateFor(classTypeExpression, subscriptionContext)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void testCheckForUnknownType() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A :
        def __init__(self):
         pass
      a = A()
      """);
    Expression unknownTypeExpression = Mockito.mock(Expression.class);
    Mockito.when(unknownTypeExpression.typeV2()).thenReturn(PythonType.UNKNOWN);

    Expression knownTypeExpression = project.lastExpression("""
      from my_file import A
      a = A()
      a
      """);

    var ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    IsSubtypeOfPredicate isSubtypeOfAPredicate = new IsSubtypeOfPredicate("my_file.A");
    IsSubtypeOfPredicate isSubtypeOfBPredicate = new IsSubtypeOfPredicate("my_file.B");

    assertThat(isSubtypeOfAPredicate.check(unknownTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isSubtypeOfBPredicate.check(knownTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(unknownTypeExpression.typeV2()).isInstanceOf(UnknownType.class);
  }
}

