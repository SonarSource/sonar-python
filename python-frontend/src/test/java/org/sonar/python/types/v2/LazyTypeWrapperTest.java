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

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.python.semantic.v2.LazyTypesContext;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.STR_TYPE;
import static org.assertj.core.api.Assertions.assertThatThrownBy;


class LazyTypeWrapperTest {


  @Test
  void lazyTypeIsResolvedWhenAccessed() {
    LazyTypesContext lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    when(lazyTypesContext.resolveLazyType(Mockito.any())).thenReturn(INT_TYPE);
    LazyType lazyType = new LazyType("random", lazyTypesContext);
    LazyTypeWrapper lazyTypeWrapper = new LazyTypeWrapper(lazyType);
    assertThat(lazyTypeWrapper.isResolved()).isFalse();
    assertThat(lazyTypeWrapper.type()).isEqualTo(INT_TYPE);
    assertThat(lazyTypeWrapper.isResolved()).isTrue();
  }

  @Test
  void resolvingNonLazyTypeThrowsException() {
    LazyTypeWrapper lazyTypeWrapper = new LazyTypeWrapper(INT_TYPE);
    assertThatThrownBy(() -> lazyTypeWrapper.resolveLazyType(STR_TYPE))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Trying to resolve an already resolved lazy type.");
  }

  @Test
  void testEquals() {
    var lazyTypeWrapper = new LazyTypeWrapper(new LazyType("random", Mockito.mock(LazyTypesContext.class)));
    assertThat(lazyTypeWrapper.equals(lazyTypeWrapper)).isTrue();
    assertThat(lazyTypeWrapper.equals(null)).isFalse();
    assertThat(lazyTypeWrapper.equals("str")).isFalse();
  }

  @Test
  void testToString() {
    var lazyTypeWrapper = new LazyTypeWrapper(new LazyType("random", Mockito.mock(LazyTypesContext.class)));
    assertThat(lazyTypeWrapper.toString()).startsWith("LazyTypeWrapper{type=");
  }
}
