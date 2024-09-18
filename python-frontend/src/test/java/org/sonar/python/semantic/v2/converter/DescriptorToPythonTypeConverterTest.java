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
package org.sonar.python.semantic.v2.converter;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.PythonType;

class DescriptorToPythonTypeConverterTest {

  @Test
  void unknownDescriptorToPythonTypeConverterTest() {
    var ctx = Mockito.mock(ConversionContext.class);
    var descriptor = Mockito.mock(Descriptor.class);
    Mockito.when(descriptor.kind()).thenReturn(null);
    var converter = new UnknownDescriptorToPythonTypeConverter();
    var type = converter.convert(ctx, descriptor);
    Assertions.assertThat(type).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void ambiguousDescriptorConversionTest() {
    var lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    var converter = new AnyDescriptorToPythonTypeConverter(lazyTypesContext);
    var descriptor = Mockito.mock(AmbiguousDescriptor.class);
    Mockito.when(descriptor.kind()).thenReturn(Descriptor.Kind.AMBIGUOUS);

    var type = converter.convert(descriptor);
    Assertions.assertThat(type).isEqualTo(PythonType.UNKNOWN);
  }



}
