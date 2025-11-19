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
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.semantic.v2.TestProject;

import static org.assertj.core.api.Assertions.assertThat;

class IsObjectOfTypePredicateTest {


  @Test
  void testCheckForObjectType() {
    var project = new TestProject();
    project.addModule("my_file.py", """
    class A :
      def __init__(self):
        pass
    class B :
      def __init__(self):
        pass
    """);
    Expression objectTypeExpression = project.lastExpression("""
      from my_file import A
      a = A()
      a
      """);

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());

    IsObjectOfTypePredicate isObjectOfTypeAPredicate = new IsObjectOfTypePredicate("my_file.A");
    IsObjectOfTypePredicate isObjectOfTypeBPredicate = new IsObjectOfTypePredicate("my_file.B");

    assertThat(isObjectOfTypeAPredicate.check(objectTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.isObjectOfType("my_file.A").isTrueFor(objectTypeExpression, ctx)).isTrue();
    assertThat(isObjectOfTypeBPredicate.check(objectTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);

    assertThat(objectTypeExpression.typeV2()).isNotInstanceOf(UnknownType.class);
  }

  @Test
  void testCheckForFunctionType() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      def func1(): pass
      """);
    Expression func1Expression = project.lastExpression("""
      from my_file import func1
      func1
      """);

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());

    IsObjectOfTypePredicate isObjectOfTypePredicateFunction1 = new IsObjectOfTypePredicate("my_file.func1");

    // isOfType is TRUE for object type only
    assertThat(isObjectOfTypePredicateFunction1.check(func1Expression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);
    assertThat(func1Expression.typeV2()).isNotInstanceOf(UnknownType.class);
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
    Expression unknownTypeExpression = project.lastExpression("""
      from my_file import a
      a
      """);
    Expression knownTypeExpression = project.lastExpression("""
      from my_file import A
      a = A()
      a
      """);

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());

    IsObjectOfTypePredicate isObjectOfTypeAPredicate = new IsObjectOfTypePredicate("my_file.A");
    IsObjectOfTypePredicate isObjectOfTypeBPredicate = new IsObjectOfTypePredicate("my_file.B");

    // Importing 'a' from another file result in UnknownType for 'a'
    assertThat(isObjectOfTypeAPredicate.check(unknownTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isObjectOfTypeBPredicate.check(knownTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(unknownTypeExpression.typeV2()).isInstanceOf(UnknownType.class);
  }
}

