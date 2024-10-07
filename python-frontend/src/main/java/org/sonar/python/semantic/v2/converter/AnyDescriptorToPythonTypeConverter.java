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

import java.util.EnumMap;
import java.util.Map;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeOrigin;

public class AnyDescriptorToPythonTypeConverter {

  private static final DescriptorToPythonTypeConverter UNKNOWN_DESCRIPTOR_CONVERTER = new UnknownDescriptorToPythonTypeConverter();
  private final Map<Descriptor.Kind, DescriptorToPythonTypeConverter> converters;
  private final LazyTypesContext lazyTypesContext;
  private final TypeOrigin typeOrigin;

  public AnyDescriptorToPythonTypeConverter(LazyTypesContext lazyTypesContext, TypeOrigin typeOrigin) {
    this.lazyTypesContext = lazyTypesContext;
    this.typeOrigin = typeOrigin;
    converters = new EnumMap<>(Map.of(
      Descriptor.Kind.CLASS, new ClassDescriptorToPythonTypeConverter(),
      Descriptor.Kind.FUNCTION, new FunctionDescriptorToPythonTypeConverter(),
      Descriptor.Kind.VARIABLE, new VariableDescriptorToPythonTypeConverter(),
      Descriptor.Kind.AMBIGUOUS, new AmbiguousDescriptorToPythonTypeConverter()
    ));

  }

  public PythonType convert(Descriptor from) {
    var ctx = new ConversionContext(lazyTypesContext, this::convert, typeOrigin);
    return convert(ctx, from);
  }

  private PythonType convert(ConversionContext ctx, Descriptor from) {
    return converterFor(from).convert(ctx, from);
  }

  private DescriptorToPythonTypeConverter converterFor(Descriptor descriptor) {
    return converters.getOrDefault(descriptor.kind(), UNKNOWN_DESCRIPTOR_CONVERTER);
  }

}
