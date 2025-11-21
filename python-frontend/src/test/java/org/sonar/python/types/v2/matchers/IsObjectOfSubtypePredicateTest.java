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
import org.sonar.python.semantic.v2.TestProject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.matchers.TypeMatchers.isObjectOfSubType;

class IsObjectOfSubtypePredicateTest {

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

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());

    IsObjectSubtypeOfPredicate isObjectOfSubtypeBPredicate = new IsObjectSubtypeOfPredicate("my_file.B");
    IsObjectSubtypeOfPredicate isObjectOfSubtypeAPredicate = new IsObjectSubtypeOfPredicate("my_file.A");
    IsObjectSubtypeOfPredicate isObjectOfSubtypeCPredicate = new IsObjectSubtypeOfPredicate("my_file.C");

    assertThat(isObjectOfSubtypeAPredicate.check(objectTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);
    assertThat(isObjectOfSubType("my_file.A").evaluateFor(objectTypeExpression, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(isObjectOfSubtypeBPredicate.check(objectTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.TRUE);

    assertThat(isObjectOfSubtypeCPredicate.check(objectTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);
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

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());
    IsObjectSubtypeOfPredicate isObjectOfSubtypeAPredicate = new IsObjectSubtypeOfPredicate("my_file.A");
    assertThat(isObjectOfSubtypeAPredicate.check(classTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);
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

    SubscriptionContext ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());

    IsObjectSubtypeOfPredicate isObjectOfSubtypeAPredicate = new IsObjectSubtypeOfPredicate("my_file.A");
    IsObjectSubtypeOfPredicate isObjectOfSubtypeBPredicate = new IsObjectSubtypeOfPredicate("my_file.B");

    assertThat(isObjectOfSubtypeAPredicate.check(unknownTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isObjectOfSubtypeBPredicate.check(knownTypeExpression.typeV2(), ctx)).isEqualTo(TriBool.FALSE);
    assertThat(unknownTypeExpression.typeV2()).isInstanceOf(UnknownType.class);
  }
}

