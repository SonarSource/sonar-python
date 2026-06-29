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
package org.sonar.python.types.v2;

import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;
import static org.sonar.python.types.v2.TypesTestUtils.FLOAT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;

class LazyUnionTypeTest {

  @Test
  void lazyUnionTypeResolvesNestedLazyTypesWhenAccessed() {
    LazyTypesContext lazyTypesContext = mock(LazyTypesContext.class);
    when(lazyTypesContext.resolveLazyType(any())).thenReturn(INT_TYPE);
    LazyType lazyType = new LazyType("random", lazyTypesContext);
    LazyUnionType lazyUnionType = (LazyUnionType) LazyUnionType.or(Set.of(lazyType, FLOAT_TYPE));
    UnionType unionType = (UnionType) lazyUnionType.resolve();
    assertThat(unionType.candidates()).containsExactlyInAnyOrder(INT_TYPE, FLOAT_TYPE);
  }

  @Test
  void flattened() {
    LazyTypesContext lazyTypesContext = spy(new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())));

    var lazyType1 = lazyTypeUnresolved("lazy1", lazyTypesContext);
    var lazyType2 = lazyTypeUnresolved("lazy2", lazyTypesContext);
    var lazyType3 = lazyTypeUnresolved("lazy3", lazyTypesContext);

    var lazyUnionType1 = LazyUnionType.or(Set.of(lazyType1, lazyType2, lazyType3));
    var lazyUnionType2 = (LazyUnionType) LazyUnionType.or(Set.of(lazyType1, lazyType2, lazyType3, lazyUnionType1));
    assertThat(lazyUnionType2.candidates()).containsExactlyInAnyOrder(lazyType1, lazyType2, lazyType3);
  }

  @Test
  void emptyLazyUnionType() {
    assertThat(LazyUnionType.or(Set.of())).isEqualTo(PythonType.UNKNOWN);
  }

  LazyType lazyTypeUnresolved(String name, LazyTypesContext lazyTypesContext) {
    var lazyType = lazyTypesContext.getOrCreateLazyType(name);
    when(lazyTypesContext.resolveLazyType(lazyType)).thenReturn(new UnknownType.UnresolvedImportType(name));
    return lazyType;
  }
}
