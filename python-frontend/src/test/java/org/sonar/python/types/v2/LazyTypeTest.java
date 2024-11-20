/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.when;
import static org.sonar.python.types.v2.TypesTestUtils.BOOL_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;

class LazyTypeTest {

  @Test
  void lazyTypeThrowExceptionsWhenInteractedWith() {
    LazyTypesContext lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    when(lazyTypesContext.resolveLazyType(Mockito.any())).thenReturn(INT_TYPE);
    LazyType lazyType = new LazyType("random", lazyTypesContext);
    assertThat(lazyType.resolve()).isEqualTo(INT_TYPE);
    assertThatThrownBy(lazyType::unwrappedType).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
    assertThatThrownBy(() -> lazyType.hasMember("__bool__")).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
    assertThatThrownBy(lazyType::name).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
    assertThatThrownBy(lazyType::instanceDisplayName).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
    assertThatThrownBy(() -> lazyType.resolveMember("__bool__")).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
    assertThatThrownBy(() -> lazyType.isCompatibleWith(BOOL_TYPE)).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
    assertThatThrownBy(lazyType::key).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
    assertThatThrownBy(lazyType::definitionLocation).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
    assertThatThrownBy(lazyType::typeSource).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
    assertThatThrownBy(lazyType::displayName).isInstanceOf(IllegalStateException.class).hasMessage("Lazy types should not be interacted with.");
  }

  @Test
  void resolutionOfLazyTypeOfMethod() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    LazyTypesContext lazyTypesContext = projectLevelTypeTable.lazyTypesContext();
    LazyType lazyType = new LazyType("calendar.Calendar.iterweekdays", lazyTypesContext);
    PythonType pythonType = lazyType.resolve();
    assertThat(pythonType).isInstanceOf(FunctionType.class);
    FunctionType functionType = (FunctionType) pythonType;
    assertThat(functionType.name()).isEqualTo("iterweekdays");
    assertThat(functionType.parameters()).hasSize(1);
  }
}
