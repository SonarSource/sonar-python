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
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.v2.TestProject;
import org.sonar.python.semantic.v2.typetable.TypeTable;

import static org.assertj.core.api.Assertions.assertThat;

class IsTypePredicateTest {

  @Test
  void testCheckForObjectTypeEquality() {
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
    TypePredicateContext predicateContext = TypePredicateContext.of(project.projectLevelTypeTable());

    IsTypePredicate isTypeAPredicate = new IsTypePredicate("my_file.A");
    IsTypePredicate isTypeBPredicate = new IsTypePredicate("my_file.B");

    assertThat(isTypeAPredicate.check(objectTypeExpression.typeV2(), predicateContext)).isEqualTo(TriBool.FALSE);
    assertThat(isTypeBPredicate.check(objectTypeExpression.typeV2(), predicateContext)).isEqualTo(TriBool.FALSE);
    
    assertThat(objectTypeExpression.typeV2()).isNotInstanceOf(UnknownType.class);
  }

  @Test
  void testCheckForClassTypeEquality() {
    var project = new TestProject();
    project.addModule("my_file.py", """
    class A :
      def __init__(self):
        pass
    class B :
      def __init__(self):
        pass
    """);
    Expression classTypeExpression = project.lastExpression("""
      from my_file import A
      A
      """);

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());
    TypePredicateContext predicateContext = TypePredicateContext.of(project.projectLevelTypeTable());

    IsTypePredicate isTypeAPredicate = new IsTypePredicate("my_file.A");
    IsTypePredicate isTypeBPredicate = new IsTypePredicate("my_file.B");


    assertThat(isTypeAPredicate.check(classTypeExpression.typeV2(), predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.isType("my_file.A").isTrueFor(classTypeExpression, ctx)).isTrue();
    assertThat(isTypeBPredicate.check(classTypeExpression.typeV2(), predicateContext)).isEqualTo(TriBool.FALSE);

    assertThat(classTypeExpression.typeV2()).isNotInstanceOf(UnknownType.class);
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
      from nonexistent import something
      something
      """);
    Expression knownTypeExpression = project.lastExpression("""
      from my_file import A
      a = A()
      a
      """);

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());
    TypePredicateContext predicateContext = TypePredicateContext.of(project.projectLevelTypeTable());

    IsTypePredicate isTypeAPredicate = new IsTypePredicate("my_file.A");
    IsTypePredicate isTypeBPredicate = new IsTypePredicate("my_file.B");

    assertThat(unknownTypeExpression.typeV2()).isInstanceOf(UnknownType.class);
    assertThat(knownTypeExpression.typeV2()).isInstanceOf(ObjectType.class);

    assertThat(isTypeAPredicate.check(unknownTypeExpression.typeV2(), predicateContext)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isTypeBPredicate.check(knownTypeExpression.typeV2(), predicateContext)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void testIsTypeThroughTypeMatchers() {
    var project = new TestProject();
    project.addModule("my_file.py", """
    class A :
      def __init__(self):
        pass
    """);
    Expression classTypeExpression = project.lastExpression("""
      from my_file import A
      A
      """);

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());

    assertThat(TypeMatchers.isType("my_file.A").isTrueFor(classTypeExpression, ctx)).isTrue();
    assertThat(TypeMatchers.isType("my_file.B").isTrueFor(classTypeExpression, ctx)).isFalse();
  }

  @Test
  void testWithFqnCollision() {
    var project = new TestProject();
    project.addModule("package/__init__.py", """
      class A :
        def __init__(self):
          pass
        def A(): ... # has FQN package.A.A
        """);
    project.addModule("package/A.py", """
      class A: # has FQN package.A.A
        def __init__(self):
          pass
      """);
    Expression classTypeExpression = project.lastExpression("""
      # because with "from X import Y" X has to be a module, python has to resolve the FQN to package/A.py
      from package.A import A
      A # will result the class A defined in package/A.py
      """);

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());

    // isType(fqn) resolves the FQN like "import package.A.A" which will prefer the class A defined in __init__.py
    assertThat(TypeMatchers.isType("package.A.A").evaluateFor(classTypeExpression, ctx)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void testCheckForSelfType() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class A:
        pass
      """);
    Expression classTypeExpression = project.lastExpression("""
      from my_file import A
      A
      """);

    var ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var selfType = SelfType.of(classTypeExpression.typeV2());
    IsTypePredicate isTypePredicate = new IsTypePredicate("my_file.A");
    
    assertThat(isTypePredicate.check(selfType, ctx)).isEqualTo(TriBool.UNKNOWN);
    
    var mockTypeTable = Mockito.mock(TypeTable.class);
    // mocking this to improve coverage, should normally never happen
    Mockito.when(mockTypeTable.getType("my_file.A")).thenReturn(selfType);

    ctx = TypePredicateContext.of(mockTypeTable);
    assertThat(isTypePredicate.check(classTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.UNKNOWN);
  }
}

