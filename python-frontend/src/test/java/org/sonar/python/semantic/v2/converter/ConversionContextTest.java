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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeOrigin;

class ConversionContextTest {

  @Test
  void lazyTypeContextTest() {
    var expectedLazyTypeContext = Mockito.mock(LazyTypesContext.class);
    var rootConverter = Mockito.mock(DescriptorToPythonTypeConverter.class);
    var ctx = new ConversionContext(expectedLazyTypeContext, rootConverter, TypeOrigin.LOCAL);
    var lazyTypesContext = ctx.lazyTypesContext();
    Assertions.assertSame(expectedLazyTypeContext, lazyTypesContext);
  }

  @Test
  void parentsTest() {
    var expectedLazyTypeContext = Mockito.mock(LazyTypesContext.class);
    var rootConverter = Mockito.mock(DescriptorToPythonTypeConverter.class);
    var ctx = new ConversionContext(expectedLazyTypeContext, rootConverter, TypeOrigin.LOCAL);
    var firstParent = Mockito.mock(PythonType.class);
    var secondParent = Mockito.mock(PythonType.class);

    Assertions.assertNull(ctx.currentParent());
    ctx.pushParent(firstParent);
    Assertions.assertSame(firstParent, ctx.currentParent());
    ctx.pushParent(secondParent);
    Assertions.assertSame(secondParent, ctx.currentParent());
    Assertions.assertSame(secondParent, ctx.pollParent());
    Assertions.assertSame(firstParent, ctx.currentParent());
    Assertions.assertSame(firstParent, ctx.pollParent());
    Assertions.assertNull(ctx.currentParent());
  }

  @Test
  void convertTest() {
    var descriptor = Mockito.mock(Descriptor.class);
    var expectedType = Mockito.mock(PythonType.class);

    var lazyTypeContext = Mockito.mock(LazyTypesContext.class);
    var rootConverter = Mockito.mock(DescriptorToPythonTypeConverter.class);
    var ctx = new ConversionContext(lazyTypeContext, rootConverter, TypeOrigin.LOCAL);

    Mockito.when(rootConverter.convert(ctx, descriptor))
      .thenReturn(expectedType);

    var type = ctx.convert(descriptor);
    Assertions.assertSame(expectedType, type);
  }

}
