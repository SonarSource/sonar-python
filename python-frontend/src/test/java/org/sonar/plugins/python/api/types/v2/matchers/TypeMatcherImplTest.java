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
package org.sonar.plugins.python.api.types.v2.matchers;

import java.util.Set;
import javax.annotation.Nullable;
import org.jspecify.annotations.NonNull;
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
import org.sonar.python.semantic.v2.typetable.TypeTable;
import org.sonar.python.types.v2.matchers.TypePredicate;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.api.types.v2.matchers.MatchersTestUtils.createTypeMatcher;

class TypeMatcherImplTest {

  TypePredicate isFunctionType = new TypePredicate() {
    @Override
    public TriBool check(@NonNull PythonType type, @Nullable TypePredicateContext ctx) {
      if (type instanceof UnknownType) {
        return TriBool.UNKNOWN;
      }
      return type instanceof FunctionType ? TriBool.TRUE : TriBool.FALSE;
    }
  };

  TypeMatcher typeMatcher = createTypeMatcher(isFunctionType);

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
  void testIsFor() {
    SubscriptionContext ctx = Mockito.mock();
    Mockito.when(ctx.typeTable()).thenReturn(Mockito.mock(TypeTable.class));
    assertThat(typeMatcher.evaluateFor(functionExpr, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(typeMatcher.evaluateFor(unknownExpr, ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeMatcher.evaluateFor(objectExpr, ctx)).isEqualTo(TriBool.FALSE);
    assertThat(typeMatcher.evaluateFor(unionWithFunctionAndObjectExpr, ctx)).isEqualTo(TriBool.FALSE);
    assertThat(typeMatcher.evaluateFor(unionWithObjectAndUnknownExpr, ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeMatcher.evaluateFor(unionWithFunctionAndUnknownExpr, ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeMatcher.evaluateFor(unionOfFunctionExpr, ctx)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testIsTrueFor() {
    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(Mockito.mock(TypeTable.class));
    assertThat(typeMatcher.isTrueFor(functionExpr, ctx)).isTrue();
    assertThat(typeMatcher.isTrueFor(unknownExpr, ctx)).isFalse();
    assertThat(typeMatcher.isTrueFor(objectExpr, ctx)).isFalse();
    assertThat(typeMatcher.isTrueFor(unionWithFunctionAndObjectExpr, ctx)).isFalse();
    assertThat(typeMatcher.isTrueFor(unionWithObjectAndUnknownExpr, ctx)).isFalse();
    assertThat(typeMatcher.isTrueFor(unionWithFunctionAndUnknownExpr, ctx)).isFalse();
    assertThat(typeMatcher.isTrueFor(unionOfFunctionExpr, ctx)).isTrue();
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

    assertThat(TypeMatchers.isObjectOfType("my_file.A").evaluateFor(objectTypeExpression, ctx)).isEqualTo(TriBool.TRUE);
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

    assertThat(TypeMatchers.isObjectOfType("my_file.func1").evaluateFor(func1Expression, ctx)).isEqualTo(TriBool.FALSE);
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

    assertThat(TypeMatchers.isObjectOfType("my_file.A").evaluateFor(unknownTypeExpression, ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(TypeMatchers.isObjectOfType("my_file.B").evaluateFor(knownTypeExpression, ctx)).isEqualTo(TriBool.UNKNOWN);
  }
}

