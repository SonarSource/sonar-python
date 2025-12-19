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
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.v2.TestProject;

import static org.assertj.core.api.Assertions.assertThat;

class IsTypeOrSuperTypeSatisfyingPredicateTest {

  @Test
  void testDirectTypeMatchWithIsTypePredicate() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A:
        pass
      """);
    Expression classTypeExpression = project.lastExpression("""
      from my_file import A
      A
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    SubscriptionContext subscriptionContext = Mockito.mock(SubscriptionContext.class);
    Mockito.when(subscriptionContext.typeTable()).thenReturn(project.projectLevelTypeTable());

    var predicate = new IsTypeOrSuperTypeSatisfyingPredicate(new IsTypePredicate("my_file.A"));

    assertThat(predicate.check(classTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.isOrExtendsType("my_file.A").evaluateFor(classTypeExpression, subscriptionContext)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testSubclassMatchWithIsTypePredicate() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A:
        pass
      class B(A):
        pass
      """);
    Expression classBExpression = project.lastExpression("""
      from my_file import B
      B
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    SubscriptionContext subscriptionContext = Mockito.mock(SubscriptionContext.class);
    Mockito.when(subscriptionContext.typeTable()).thenReturn(project.projectLevelTypeTable());

    var predicateA = new IsTypeOrSuperTypeSatisfyingPredicate(new IsTypePredicate("my_file.A"));
    var predicateB = new IsTypeOrSuperTypeSatisfyingPredicate(new IsTypePredicate("my_file.B"));

    assertThat(predicateA.check(classBExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
    assertThat(predicateB.check(classBExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.isOrExtendsType("my_file.A").evaluateFor(classBExpression, subscriptionContext)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testMultiLevelInheritanceWithIsTypePredicate() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A:
        pass
      class B(A):
        pass
      class C(B):
        pass
      """);
    Expression classCExpression = project.lastExpression("""
      from my_file import C
      C
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var predicateA = new IsTypeOrSuperTypeSatisfyingPredicate(new IsTypePredicate("my_file.A"));
    var predicateB = new IsTypeOrSuperTypeSatisfyingPredicate(new IsTypePredicate("my_file.B"));
    var predicateC = new IsTypeOrSuperTypeSatisfyingPredicate(new IsTypePredicate("my_file.C"));

    assertThat(predicateA.check(classCExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
    assertThat(predicateB.check(classCExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
    assertThat(predicateC.check(classCExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testNoMatchWithIsTypePredicate() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A:
        pass
      class B:
        pass
      """);
    Expression classBExpression = project.lastExpression("""
      from my_file import B
      B
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var predicateA = new IsTypeOrSuperTypeSatisfyingPredicate(new IsTypePredicate("my_file.A"));

    assertThat(predicateA.check(classBExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void testWithUnknownSuperType() {
    var project = new TestProject();
    var classAExpression = project.lastExpression("""
      import nonexistent
      class A(nonexistent.Nonexistent):
        pass
      A
      """);

    assertThat(classAExpression.typeV2()).isInstanceOf(ClassType.class);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var predicateA = new IsTypeOrSuperTypeSatisfyingPredicate(new IsTypePredicate("nonexistent.Nonexistent"));

    assertThat(predicateA.check(classAExpression.typeV2(), ctx)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void testDirectFQNMatchWithHasFQNPredicate() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A:
        pass
      """);
    Expression classTypeExpression = project.lastExpression("""
      from my_file import A
      A
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    SubscriptionContext subscriptionContext = Mockito.mock(SubscriptionContext.class);
    Mockito.when(subscriptionContext.typeTable()).thenReturn(project.projectLevelTypeTable());

    var predicate = new IsTypeOrSuperTypeSatisfyingPredicate(new HasFQNPredicate("my_file.A"));

    assertThat(predicate.check(classTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.isTypeOrSuperTypeWithFQN("my_file.A").evaluateFor(classTypeExpression, subscriptionContext)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testSubclassFQNMatchWithHasFQNPredicate() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A:
        pass
      class B(A):
        pass
      """);
    Expression classBExpression = project.lastExpression("""
      from my_file import B
      B
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var predicateA = new IsTypeOrSuperTypeSatisfyingPredicate(new HasFQNPredicate("my_file.A"));

    assertThat(predicateA.check(classBExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testUnknownType() {
    var project = new TestProject();
    Expression unknownTypeExpression = Mockito.mock(Expression.class);
    Mockito.when(unknownTypeExpression.typeV2()).thenReturn(PythonType.UNKNOWN);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var predicate = new IsTypeOrSuperTypeSatisfyingPredicate(new IsTypePredicate("my_file.A"));

    assertThat(predicate.check(unknownTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(unknownTypeExpression.typeV2()).isInstanceOf(UnknownType.class);
  }
}
