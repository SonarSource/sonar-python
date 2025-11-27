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
package org.sonar.python.semantic.v2.types;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.python.semantic.v2.TestProject;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

import static org.assertj.core.api.Assertions.assertThat;

class TypeInferenceMatchersTest {

  @Test
  void test() {
    var project = new TestProject();
    project.addModule("test/__init__.py", "");
    project.addModule("test/my_file.py", "class A: pass");

    Expression objectTypeExpression = project.lastExpression("""
      from test.my_file import A
      a = A()
      a
      """);

    var ctx = TypePredicateContext.of(project.projectLevelTypeTable());

    var result = TypeInferenceMatcher.of(
      TypeInferenceMatchers.isObjectOfType("test.my_file.A")
    ).evaluate(objectTypeExpression.typeV2(), ctx);

    assertThat(result).isEqualTo(TriBool.TRUE);
  }
}
