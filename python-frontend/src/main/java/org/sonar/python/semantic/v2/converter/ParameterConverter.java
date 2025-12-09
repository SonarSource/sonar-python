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

import java.util.Optional;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.python.index.FunctionDescriptor;

public class ParameterConverter {

  private final TypeAnnotationToPythonTypeConverter typeAnnotationConverter = new TypeAnnotationToPythonTypeConverter();

  public ParameterV2 convert(ConversionContext ctx, FunctionDescriptor.Parameter parameter) {
    // Prefer TypeAnnotationDescriptor if available, otherwise use legacy string-based approach
    var type = getTypeFromDescriptor(ctx, parameter)
      .or(() -> getTypeFromFqn(ctx, parameter))
      .orElseGet(() -> PythonType.UNKNOWN);

    return new ParameterV2(parameter.name(),
      TypeWrapper.of(type),
      parameter.hasDefaultValue(),
      parameter.isKeywordOnly(),
      parameter.isPositionalOnly(),
      parameter.isKeywordVariadic(),
      parameter.isPositionalVariadic(),
      parameter.location());
  }

  private Optional<PythonType> getTypeFromDescriptor(ConversionContext ctx, FunctionDescriptor.Parameter parameter) {
    return Optional.ofNullable(parameter.descriptor())
      .map(typeAnnotationDescriptor -> typeAnnotationConverter.convert(ctx, typeAnnotationDescriptor));
  }

  private static Optional<PythonType> getTypeFromFqn(ConversionContext ctx, FunctionDescriptor.Parameter parameter) {
    return Optional.ofNullable(parameter.annotatedType())
      .map(fqn -> (PythonType) ctx.lazyTypesContext().getOrCreateLazyType(fqn));
  }

}
