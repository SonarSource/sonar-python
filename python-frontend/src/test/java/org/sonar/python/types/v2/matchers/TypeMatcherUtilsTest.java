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

import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.platform.commons.annotation.Testable;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.python.semantic.v2.TestProject;

import static org.assertj.core.api.Assertions.assertThat;

@Testable
class TypeMatcherUtilsTest {
  
  @Test
  void testAllCollectorBehavior() {
    var project = new TestProject();
    project.addModule("package/__init__.py", "");
    project.addModule("package/my_file.py", """
    class A :
      def __init__(self):
        pass

      def foo(self):
        pass
      
      def bar(self):
        pass
    """);

    Expression expression = project.lastExpression("""
      from package.my_file import A
      a = A()
      a
      """);


    TypeMatcher allMatcher1 = List.of("bar", "foo").stream()
      .map(TypeMatchers::hasMember)
      .collect(TypeMatcherUtils.allCollector());
    assertThat(allMatcher1.evaluateFor(expression, Mockito.mock())).isEqualTo(TriBool.TRUE);

    var allMatcher2 = TypeMatchers.all(Stream.of("bar", "foo").map(TypeMatchers::hasMember));
    assertThat(allMatcher2.evaluateFor(expression, Mockito.mock())).isEqualTo(TriBool.TRUE);
  }

  @Test
  void testAnyCollectorBehavior() {
    var project = new TestProject();
    project.addModule("my_file.py", """
    class A :
      pass
    class B :
      pass
    """);
    Expression expression = project.lastExpression("""
    from my_file import A
    a = A()
    a
    """);


    TypeMatcher anyMatcher1 = List.of("my_file.A", "my_file.B").stream()
      .map(TypeMatchers::isObjectOfType)
      .collect(TypeMatcherUtils.anyCollector());

    var ctx = Mockito.mock(SubscriptionContext.class);
    Mockito.when(ctx.typeTable()).thenReturn(project.projectLevelTypeTable());

    assertThat(anyMatcher1.evaluateFor(expression, ctx)).isEqualTo(TriBool.TRUE);

    var anyMatcher2 = TypeMatchers.any(Stream.of("my_file.A", "my_file.B").map(TypeMatchers::isObjectOfType));
    assertThat(anyMatcher2.evaluateFor(expression, ctx)).isEqualTo(TriBool.TRUE);
  }
}
