/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.semantic.v2;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.v2.types.ClassDecoratorTypes;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;

class ClassDecoratorTypesTest {
  @ParameterizedTest
  @ValueSource(strings = {"%s", "(%s)", "(((%s)))"})
  void parentheses_preserve_decorator_values_and_factory_callees(String syntax) {
    var definition = TypesTestUtils.lastClassDef("""
      from typing import final
      from dataclasses import dataclass
      @%s
      @%s
      @%s
      class C: pass
      """.formatted(syntax.formatted("final"), syntax.formatted("dataclass()"), syntax.formatted("unknown_factory()")));
    var decorators = ((ClassType) definition.name().typeV2()).decorators();
    assertThat(decorators).hasSize(3);
    assertThat(decorators.get(0).expressionType().type()).isSameAs(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE.getType("typing.final"));
    assertThat(decorators.get(0).calleeType()).isEmpty();
    assertThat(decorators.get(1).calleeType()).hasValueSatisfying(callee ->
      assertThat(callee.type()).isSameAs(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE.getType("dataclasses.dataclass")));
    assertThat(decorators.get(2).expressionType().type()).isEqualTo(PythonType.UNKNOWN);
    assertThat(decorators.get(2).calleeType()).hasValueSatisfying(callee -> assertThat(callee.type()).isEqualTo(PythonType.UNKNOWN));
  }

  @Test
  void early_reads_do_not_freeze_source_decorator_slots_to_unknown() {
    var source = PythonTestUtils.parseWithoutSymbols("@marker\n@factory()\nclass C: pass");
    ClassDef definition = PythonTestUtils.getLastDescendant(source, t -> t.is(Tree.Kind.CLASSDEF));
    var decorators = ClassDecoratorTypes.deferred(definition);
    var type = new ClassTypeBuilder("C", "mod.C").withDecorators(decorators).build();
    ((NameImpl) definition.name()).typeV2(type);
    var earlyReference = type.decorators().get(0).expressionType();
    assertThat(earlyReference.type()).isEqualTo(PythonType.UNKNOWN);
    assertThat(type.decorators().get(1).calleeType().orElseThrow().type()).isEqualTo(PythonType.UNKNOWN);

    // Simulate the expression updates made by inference before metadata completion.
    var marker = TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE.getType("typing.final");
    ((NameImpl) definition.decorators().get(0).expression()).typeV2(marker);
    var call = (CallExpression) definition.decorators().get(1).expression();
    ((NameImpl) call.callee()).typeV2(marker);
    ClassDecoratorTypes.completeScope(source);

    assertThat(earlyReference.type()).isEqualTo(marker);
    assertThat(type.decorators().get(1).expressionType().type()).isEqualTo(PythonType.UNKNOWN);
    assertThat(type.decorators().get(1).calleeType().orElseThrow().type()).isEqualTo(marker);
  }
}
