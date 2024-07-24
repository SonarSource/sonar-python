/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.semantic.v2;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.PythonType;

class ProjectLevelTypeTableTest {

  @Test
  void getTypeTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable, new TypeShed(symbolTable));

    var callableClassType = table.getType("typing.Callable");
    Assertions.assertThat(callableClassType).isNotNull().isInstanceOf(ClassType.class);

    var typingModuleType = table.getType("typing");
    Assertions.assertThat(typingModuleType).isNotNull().isInstanceOf(ModuleType.class);
    Assertions.assertThat(typingModuleType.resolveMember("Callable")).isPresent().containsSame(callableClassType);

    Assertions.assertThat(table.getType("typing.Callable.something")).isSameAs(PythonType.UNKNOWN);
  }
}
