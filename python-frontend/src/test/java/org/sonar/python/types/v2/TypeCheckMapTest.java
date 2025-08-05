/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.types.v2;

import java.util.Map;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;

class TypeCheckMapTest {
  @Test
  void test() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    var intBuilder = new TypeCheckBuilder(table).isTypeWithFqn("int");
    var fooBuilder = new TypeCheckBuilder(table).isTypeWithFqn("mod.foo");
    var map = TypeCheckMap.ofEntries(
      Map.entry(intBuilder, 1),
      Map.entry(fooBuilder, 2)
    );

    var intClassType = table.getType("int");
    var strClassType = table.getType("str");
    var fooClassType = new ClassTypeBuilder("foo", "mod.foo").build();

    Assertions.assertThat(map.getForType(intClassType)).isEqualTo(1);
    Assertions.assertThat(map.getForType(strClassType)).isNull();
    Assertions.assertThat(map.getForType(fooClassType)).isEqualTo(2);
    Assertions.assertThat(map.getForType(PythonType.UNKNOWN)).isNull();
    Assertions.assertThat(map.containsForType(intClassType)).isTrue();
    Assertions.assertThat(map.containsForType(strClassType)).isFalse();
    Assertions.assertThat(map.containsForType(fooClassType)).isTrue();
  }
}
