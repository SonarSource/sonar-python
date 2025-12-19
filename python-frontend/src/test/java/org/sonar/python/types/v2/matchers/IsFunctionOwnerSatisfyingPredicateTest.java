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
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.v2.TestProject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;

class IsFunctionOwnerSatisfyingPredicateTest {

  @Test
  void testFunctionWithOwnerMatchingPredicate() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class MyClass:
        def my_method(self):
          pass
      """);
    Expression methodExpression = project.lastExpression("""
      from my_file import MyClass
      MyClass.my_method
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var predicate = new IsFunctionOwnerSatisfyingPredicate(
      new IsTypeOrSuperTypeSatisfyingPredicate(
        new IsTypePredicate("my_file.MyClass")
      )
    );

    assertThat(predicate.check(methodExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testFunctionWithOwnerNotMatchingPredicate() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class ClassA:
        def method_a(self):
          pass
      class ClassB:
        def method_b(self):
          pass
      """);
    Expression methodExpression = project.lastExpression("""
      from my_file import ClassB
      ClassB.method_b
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var predicate = new IsFunctionOwnerSatisfyingPredicate(
      new IsTypeOrSuperTypeSatisfyingPredicate(
        new IsTypePredicate("my_file.ClassA")
      )
    );

    assertThat(predicate.check(methodExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void testNonFunctionType() {
    var project = new TestProject();
    project.addModule("my_file.py", """
      class MyClass:
        pass
      """);
    Expression classExpression = project.lastExpression("""
      from my_file import MyClass
      MyClass
      """);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var predicate = new IsFunctionOwnerSatisfyingPredicate(
      new IsTypePredicate("my_file.MyClass")
    );

    assertThat(predicate.check(classExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void testUnknownType() {
    var project = new TestProject();
    Expression unknownTypeExpression = Mockito.mock(Expression.class);
    Mockito.when(unknownTypeExpression.typeV2()).thenReturn(PythonType.UNKNOWN);

    TypePredicateContext ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var predicate = new IsFunctionOwnerSatisfyingPredicate(new IsTypePredicate("my_file.A"));

    assertThat(predicate.check(unknownTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(unknownTypeExpression.typeV2()).isInstanceOf(UnknownType.class);
  }

  @Test
  void testTopLevelFunction() {
    var root = parseAndInferTypes("""
      def foo(): pass
      """);
    FunctionDef functionDef = PythonTestUtils.getFirstDescendant(root, t -> t.is(Tree.Kind.FUNCDEF));
    assertThat(functionDef.name().typeV2()).isInstanceOf(FunctionType.class);

    TypePredicateContext typePredicateCtx = TypePredicateContext.of(PROJECT_LEVEL_TYPE_TABLE);
    var predicate = new IsFunctionOwnerSatisfyingPredicate((type, ctx) -> TriBool.TRUE);
    assertThat(predicate.check(functionDef.name().typeV2(), typePredicateCtx)).isEqualTo(TriBool.FALSE);
  }

}
