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
package org.sonar.python.semantic.v2.converter;

import java.util.Set;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.LazyUnionType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

class AmbiguousDescriptorToPythonTypeConverterTest {
  @Test
  void unsupportedClassTest() {
    var ctx = Mockito.mock(ConversionContext.class);
    var descriptor = Mockito.mock(ClassDescriptor.class);
    var converter = new AmbiguousDescriptorToPythonTypeConverter();
    Assertions.assertThatThrownBy(() -> converter.convert(ctx, descriptor))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessage("Unsupported Descriptor");
  }

  @Test
  void testDescriptorWithLazyTypes() {
    var descriptor = Mockito.mock(Descriptor.class);
    when(descriptor.kind()).thenReturn(Descriptor.Kind.VARIABLE);

    var descriptor2 = Mockito.mock(Descriptor.class);
    when(descriptor2.kind()).thenReturn(Descriptor.Kind.VARIABLE);

    var lazyTypeContext = Mockito.mock(LazyTypesContext.class);
    var lazyType1 = new LazyType("lazy1", lazyTypeContext);
    var lazyType2 = new LazyType("lazy2", lazyTypeContext);

    var ctx = Mockito.mock(ConversionContext.class);
    when(ctx.convert(descriptor)).thenReturn(lazyType1);
    when(ctx.convert(descriptor2)).thenReturn(lazyType2);

    var converter = new AmbiguousDescriptorToPythonTypeConverter();
    var type = converter.convert(ctx, new AmbiguousDescriptor("ambiguous", "ambiguous", Set.of(descriptor, descriptor2)));

    assertThat(type)
      .isInstanceOf(LazyUnionType.class);
  }
}
