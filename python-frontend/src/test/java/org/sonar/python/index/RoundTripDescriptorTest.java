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
package org.sonar.python.index;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.python.semantic.v2.TestProject;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.InstanceOfAssertFactories.type;

class RoundTripDescriptorTest {

  @Test
  void testRoundTripFunctionDescriptorWithAttributes() {
    var project = new TestProject();
    var intType = project.projectLevelTypeTable().getType("builtins.int");

    project.addModule("test_module.py", """
      def foo(param1: list[int]) -> list[str]:
        pass
      """);

    var fooExpr = project.lastExpression("""
      from test_module import foo
      foo
      """);

    FunctionType functionType = assertThat(fooExpr.typeV2())
      .asInstanceOf(type(FunctionType.class))
      .actual();
    

    ParameterV2 parameter = assertThat(functionType.parameters())
      .element(0)
      .actual();

    assertThat(parameter.declaredType().type())
      .isInstanceOfSatisfying(ObjectType.class, objType -> assertThat(objType.attributes())
        .element(0)
        .is(TypesTestUtils.objectTypeOf(intType)));
  }
}
