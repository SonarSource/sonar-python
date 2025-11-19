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

import java.util.Set;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.semantic.v2.TestProject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.matchers.TypeMatcher.extractCandidates;

class TypeMatcherTest {

  TypePredicate isFunctionType = new TypePredicate() {
    @Override
    public TriBool check(@NonNull PythonType type, @Nullable SubscriptionContext ctx) {
      if (type instanceof UnknownType) {
        return TriBool.UNKNOWN;
      }
      return type instanceof FunctionType ? TriBool.TRUE : TriBool.FALSE;
    }
  };

  TypeMatcher typeMatcher = new TypeMatcher(isFunctionType);

  PythonType functionType = Mockito.mock(FunctionType.class);
  PythonType functionType2 = Mockito.mock(FunctionType.class);
  PythonType objectType = Mockito.mock(ObjectType.class);
  PythonType objectType2 = Mockito.mock(ObjectType.class);
  PythonType unknownType = Mockito.mock(UnknownType.UnknownTypeImpl.class);
  PythonType unionWithFunctionAndUnknown = UnionType.or(Set.of(functionType, unknownType));
  PythonType unionWithObjectAndUnknown = UnionType.or(Set.of(objectType, unknownType));
  PythonType unionWithFunctionAndObject = UnionType.or(Set.of(functionType, objectType));
  PythonType unionOfFunctions = UnionType.or(Set.of(functionType, functionType2));
  PythonType unionOfObjects = UnionType.or(Set.of(objectType, objectType2));

  Expression functionExpr = Mockito.mock(Expression.class);
  Expression unknownExpr = Mockito.mock(Expression.class);
  Expression objectExpr = Mockito.mock(Expression.class);
  Expression unionWithFunctionAndUnknownExpr = Mockito.mock(Expression.class);
  Expression unionWithObjectAndUnknownExpr = Mockito.mock(Expression.class);
  Expression unionWithFunctionAndObjectExpr = Mockito.mock(Expression.class);
  Expression unionOfFunctionExpr = Mockito.mock(Expression.class);
  Expression unionOfObjectExpr = Mockito.mock(Expression.class);

  @BeforeEach
  void prepare() {
    functionType = Mockito.mock(FunctionType.class);
    functionType2 = Mockito.mock(FunctionType.class);
    objectType = Mockito.mock(ObjectType.class);
    objectType2 = Mockito.mock(ObjectType.class);
    unknownType = Mockito.mock(UnknownType.UnknownTypeImpl.class);
    unionWithFunctionAndUnknown = UnionType.or(Set.of(functionType, unknownType));
    unionWithObjectAndUnknown = UnionType.or(Set.of(objectType, unknownType));
    unionWithFunctionAndObject = UnionType.or(Set.of(functionType, objectType));
    unionOfFunctions = UnionType.or(Set.of(functionType, functionType2));
    unionOfObjects = UnionType.or(Set.of(objectType, objectType2));

    functionExpr = Mockito.mock(Expression.class);
    unknownExpr = Mockito.mock(Expression.class);
    objectExpr = Mockito.mock(Expression.class);
    unionWithFunctionAndUnknownExpr = Mockito.mock(Expression.class);
    unionWithObjectAndUnknownExpr = Mockito.mock(Expression.class);
    unionWithFunctionAndObjectExpr = Mockito.mock(Expression.class);
    unionOfFunctionExpr = Mockito.mock(Expression.class);
    unionOfObjectExpr = Mockito.mock(Expression.class);

    Mockito.when(unknownExpr.typeV2()).thenReturn(unknownType);
    Mockito.when(objectExpr.typeV2()).thenReturn(objectType);
    Mockito.when(functionExpr.typeV2()).thenReturn(functionType);
    Mockito.when(unionWithFunctionAndUnknownExpr.typeV2()).thenReturn(unionWithFunctionAndUnknown);
    Mockito.when(unionWithObjectAndUnknownExpr.typeV2()).thenReturn(unionWithObjectAndUnknown);
    Mockito.when(unionWithFunctionAndObjectExpr.typeV2()).thenReturn(unionWithFunctionAndObject);
    Mockito.when(unionOfFunctionExpr.typeV2()).thenReturn(unionOfFunctions);
    Mockito.when(unionOfObjectExpr.typeV2()).thenReturn(unionOfObjects);
  }

  @Test
  void testExtractCandidates() {
    Set<PythonType> objectCandidates = extractCandidates(objectType);
    assertThat(objectCandidates).hasSize(1).first().isEqualTo(objectType);

    Set<PythonType> functionCandidates = extractCandidates(functionType);
    assertThat(functionCandidates).hasSize(1).first().isEqualTo(functionType);

    PythonType unionType = UnionType.or(Set.of(objectType, functionType));
    Set<PythonType> unionCandidates = extractCandidates(unionType);
    assertThat(unionCandidates).hasSize(2).contains(objectType, functionType);
  }

  @Test
  void testIsFor() {
    assertThat(typeMatcher.isFor(functionExpr, null)).isEqualTo(TriBool.TRUE);
    assertThat(typeMatcher.isFor(unknownExpr, null)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeMatcher.isFor(objectExpr, null)).isEqualTo(TriBool.FALSE);
    assertThat(typeMatcher.isFor(unionWithFunctionAndObjectExpr, null)).isEqualTo(TriBool.FALSE);
    assertThat(typeMatcher.isFor(unionWithObjectAndUnknownExpr, null)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeMatcher.isFor(unionWithFunctionAndUnknownExpr, null)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeMatcher.isFor(unionOfFunctionExpr, null)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testCanBeFor() {
    assertThat(typeMatcher.canBeFor(functionExpr, null)).isEqualTo(TriBool.TRUE);
    assertThat(typeMatcher.canBeFor(unknownExpr, null)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeMatcher.canBeFor(objectExpr, null)).isEqualTo(TriBool.FALSE);
    assertThat(typeMatcher.canBeFor(unionWithFunctionAndObjectExpr, null)).isEqualTo(TriBool.TRUE);
    assertThat(typeMatcher.canBeFor(unionWithObjectAndUnknownExpr, null)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeMatcher.canBeFor(unionWithFunctionAndUnknownExpr, null)).isEqualTo(TriBool.TRUE);
    assertThat(typeMatcher.canBeFor(unionOfObjectExpr, null)).isEqualTo(TriBool.FALSE);
  }


  @Test
  void testIsTrueFor() {
    assertThat(typeMatcher.isTrueFor(functionExpr, null)).isTrue();
    assertThat(typeMatcher.isTrueFor(unknownExpr, null)).isFalse();
    assertThat(typeMatcher.isTrueFor(objectExpr, null)).isFalse();
    assertThat(typeMatcher.isTrueFor(unionWithFunctionAndObjectExpr, null)).isFalse();
    assertThat(typeMatcher.isTrueFor(unionWithObjectAndUnknownExpr, null)).isFalse();
    assertThat(typeMatcher.isTrueFor(unionWithFunctionAndUnknownExpr, null)).isFalse();
    assertThat(typeMatcher.isTrueFor(unionOfFunctionExpr, null)).isTrue();
  }

  @Test
  void testCanBeTrueFor() {
    assertThat(typeMatcher.canBeTrueFor(functionExpr, null)).isTrue();
    assertThat(typeMatcher.canBeTrueFor(unknownExpr, null)).isFalse();
    assertThat(typeMatcher.canBeTrueFor(objectExpr, null)).isFalse();
    assertThat(typeMatcher.canBeTrueFor(unionWithFunctionAndObjectExpr, null)).isTrue();
    assertThat(typeMatcher.canBeTrueFor(unionWithObjectAndUnknownExpr, null)).isFalse();
    assertThat(typeMatcher.canBeTrueFor(unionWithFunctionAndUnknownExpr, null)).isTrue();
    assertThat(typeMatcher.canBeTrueFor(unionOfObjectExpr, null)).isFalse();
  }

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

    assertThat(TypeMatchers.isObjectOfType("my_file.A").isFor(objectTypeExpression, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.isObjectOfType("my_file.B").isTrueFor(objectTypeExpression, ctx)).isFalse();

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

    assertThat(TypeMatchers.isObjectOfType("my_file.func1").isFor(func1Expression, ctx)).isEqualTo(TriBool.FALSE);
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

    assertThat(unknownTypeExpression.typeV2()).isInstanceOf(UnknownType.class);
    assertThat(knownTypeExpression.typeV2()).isInstanceOf(ObjectType.class);

    assertThat(TypeMatchers.isObjectOfType("my_file.A").isFor(unknownTypeExpression, ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TypeMatchers.isObjectOfType("my_file.B").isFor(knownTypeExpression, ctx)).isEqualTo(TriBool.UNKNOWN);
  }
}

