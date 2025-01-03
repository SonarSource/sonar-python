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

import java.util.Set;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;
import static org.sonar.python.types.v2.TypesTestUtils.FLOAT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;

class LazyUnionTypeTest {

  @Test
  void lazyUnionTypeResolvesNestedLazyTypesWhenAccessed() {
    LazyTypesContext lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    when(lazyTypesContext.resolveLazyType(Mockito.any())).thenReturn(INT_TYPE);
    LazyType lazyType = new LazyType("random", lazyTypesContext);
    LazyUnionType lazyUnionType = new LazyUnionType(Set.of(lazyType, FLOAT_TYPE));
    UnionType unionType = (UnionType) lazyUnionType.resolve();
    assertThat(unionType.candidates()).containsExactlyInAnyOrder(INT_TYPE, FLOAT_TYPE);
  }

  @Test
  void flattened() {
    LazyTypesContext lazyTypesContext = Mockito.spy(new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())));

    var lazyType1 = lazyTypeUnresolved("lazy1", lazyTypesContext);
    var lazyType2 = lazyTypeUnresolved("lazy2", lazyTypesContext);
    var lazyType3 = lazyTypeUnresolved("lazy3", lazyTypesContext);

    var lazyUnionType1 = new LazyUnionType(Set.of(lazyType1, lazyType2, lazyType3));
    var lazyUnionType2 = new LazyUnionType(Set.of(lazyType1, lazyType2, lazyType3, lazyUnionType1));
    assertThat(lazyUnionType2.candidates()).containsExactlyInAnyOrder(lazyType1, lazyType2, lazyType3);
  }

  LazyType lazyTypeUnresolved(String name, LazyTypesContext lazyTypesContext) {
    var lazyType = lazyTypesContext.getOrCreateLazyType(name);
    when(lazyTypesContext.resolveLazyType(lazyType)).thenReturn(new UnknownType.UnresolvedImportType(name));
    return lazyType;
  }
}
