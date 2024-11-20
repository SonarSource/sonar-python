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
package org.sonar.python.semantic.v2.converter;

import java.util.List;
import java.util.Optional;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeUtils;
import org.sonar.python.types.v2.TypeWrapper;

public class FunctionDescriptorToPythonTypeConverter implements DescriptorToPythonTypeConverter {

  private final ParameterConverter parameterConverter;
  private final TypeAnnotationToPythonTypeConverter typeAnnotationConverter;

  public FunctionDescriptorToPythonTypeConverter() {
    parameterConverter = new ParameterConverter();
    typeAnnotationConverter = new TypeAnnotationToPythonTypeConverter();
  }

  public PythonType convert(ConversionContext ctx, FunctionDescriptor from) {
    var parameters = from.parameters()
      .stream()
      .map(parameter -> parameterConverter.convert(ctx, parameter))
      .toList();

    PythonType returnType = Optional.ofNullable(from.typeAnnotationDescriptor())
      .map(typeAnnotation -> typeAnnotationConverter.convert(ctx, typeAnnotation))
      .map(TypeUtils::ensureWrappedObjectType)
      .orElse(PythonType.UNKNOWN);

    var decorators = from.decorators()
      .stream()
      .map(ctx.lazyTypesContext()::getOrCreateLazyType)
      .map(TypeWrapper::of)
      .toList();

    var typeOrigin = ctx.typeOrigin();

    var hasVariadicParameter = hasVariadicParameter(parameters);

    var toBuilder = new FunctionTypeBuilder()
      .withOwner(ctx.currentParent())
      .withName(from.name())
      .withFullyQualifiedName(from.fullyQualifiedName())
      .withParameters(parameters)
      .withDecorators(decorators)
      .withReturnType(returnType)
      .withTypeOrigin(typeOrigin)
      .withAsynchronous(from.isAsynchronous())
      .withHasDecorators(from.hasDecorators())
      .withInstanceMethod(from.isInstanceMethod())
      .withHasVariadicParameter(hasVariadicParameter)
      .withDefinitionLocation(from.definitionLocation());

    return toBuilder.build();
  }

  private static boolean hasVariadicParameter(List<ParameterV2> parameters) {
    return parameters.stream()
      .anyMatch(p -> p.isKeywordVariadic() || p.isPositionalVariadic());
  }

  @Override
  public PythonType convert(ConversionContext ctx, Descriptor from) {
    if (from instanceof FunctionDescriptor functionDescriptor) {
      return convert(ctx, functionDescriptor);
    }
    throw new IllegalArgumentException("Unsupported Descriptor");
  }
}
