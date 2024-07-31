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
package org.sonar.python.types.v2;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.SymbolsModuleTypeProvider;
import org.sonar.python.semantic.v2.TypeShed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;
import static org.sonar.python.types.v2.TypesTestUtils.BOOL_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;

class LazyTypeTest {

  @Test
  void lazyTypeResolvedWhenInteractedWith() {
    SymbolsModuleTypeProvider symbolsModuleTypeProvider = Mockito.mock(SymbolsModuleTypeProvider.class);
    when(symbolsModuleTypeProvider.resolveLazyType(Mockito.any())).thenReturn(INT_TYPE);
    LazyType lazyType = new LazyType("random", symbolsModuleTypeProvider);
    assertThat(lazyType.resolve()).isEqualTo(INT_TYPE);
    assertThat(lazyType.unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(lazyType.hasMember("__bool__")).isEqualTo(TriBool.UNKNOWN);
    assertThat(new ObjectType(lazyType, TypeSource.EXACT).hasMember("__bool__")).isEqualTo(TriBool.TRUE);
    assertThat(lazyType.name()).isEqualTo(INT_TYPE.name());
    assertThat(lazyType.instanceDisplayName()).isEqualTo(INT_TYPE.instanceDisplayName());
    assertThat(lazyType.resolveMember("__bool__")).isEqualTo(INT_TYPE.resolveMember("__bool__"));
    assertThat(lazyType.isCompatibleWith(BOOL_TYPE)).isEqualTo(INT_TYPE.isCompatibleWith(BOOL_TYPE));
    assertThat(lazyType.key()).isEqualTo(INT_TYPE.key());
    assertThat(lazyType.definitionLocation()).isEqualTo(INT_TYPE.definitionLocation());
    assertThat(lazyType.typeSource()).isEqualTo(INT_TYPE.typeSource());
    assertThat(lazyType.displayName()).contains("type");

    TypeCheckBuilder typeCheckBuilder = new TypeCheckBuilder(PROJECT_LEVEL_TYPE_TABLE);
    assertThat(typeCheckBuilder.isBuiltinWithName("int").check(lazyType)).isEqualTo(TriBool.TRUE);
    assertThat(TypeUtils.resolved(lazyType)).isEqualTo(INT_TYPE);
  }

  @Test
  void resolutionOfLazyTypeOfMethod() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    TypeShed typeShed = new TypeShed(projectLevelSymbolTable);
    SymbolsModuleTypeProvider symbolsModuleTypeProvider = new SymbolsModuleTypeProvider(projectLevelSymbolTable, typeShed);
    LazyType lazyType = new LazyType("calendar.Calendar.iterweekdays", symbolsModuleTypeProvider);
    PythonType pythonType = lazyType.resolve();
    assertThat(pythonType).isInstanceOf(FunctionType.class);
    FunctionType functionType = (FunctionType) pythonType;
    assertThat(functionType.name()).isEqualTo("iterweekdays");
    assertThat(functionType.parameters()).hasSize(1);
  }
}
