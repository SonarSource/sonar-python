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
package org.sonar.python.semantic.v2.typeshed;

import java.util.Collection;
import java.util.Optional;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class FunctionSymbolToDescriptorConverter {

  private final ParameterSymbolToDescriptorConverter parameterConverter;
  private final TypeSymbolToDescriptorConverter typeConverter;

  public FunctionSymbolToDescriptorConverter() {
    parameterConverter = new ParameterSymbolToDescriptorConverter();
    typeConverter = new TypeSymbolToDescriptorConverter();
  }

  public FunctionDescriptor convert(SymbolsProtos.FunctionSymbol functionSymbol) {
    return convert(functionSymbol, false);
  }

  public FunctionDescriptor convert(SymbolsProtos.FunctionSymbol functionSymbol, boolean isParentIsAClass) {
    var fullyQualifiedName = TypeShedUtils.normalizedFqn(functionSymbol.getFullyQualifiedName());
    TypeAnnotationDescriptor typeAnnotationDescriptor = null;
    if (functionSymbol.hasReturnAnnotation()) {
      SymbolsProtos.Type returnAnnotation = functionSymbol.getReturnAnnotation();
      typeAnnotationDescriptor = typeConverter.convert(returnAnnotation);
    }
    String returnType = TypeShedUtils.getTypesNormalizedFqn(functionSymbol.getReturnAnnotation());
    var decorators = Optional.of(functionSymbol)
      .map(SymbolsProtos.FunctionSymbol::getResolvedDecoratorNamesList)
      .stream()
      .flatMap(Collection::stream)
      .map(TypeShedUtils::normalizedFqn)
      .toList();
    var parameters = functionSymbol.getParametersList().stream()
      .map(parameterConverter::convert)
      .toList();
    var isInstanceMethod = isParentIsAClass && !functionSymbol.getIsStatic() && !functionSymbol.getIsClassMethod();
    return new FunctionDescriptor.FunctionDescriptorBuilder()
      .withName(functionSymbol.getName())
      .withFullyQualifiedName(fullyQualifiedName)
      .withIsAsynchronous(functionSymbol.getIsAsynchronous())
      .withIsInstanceMethod(isInstanceMethod)
      .withHasDecorators(functionSymbol.getHasDecorators())
      .withAnnotatedReturnTypeName(returnType)
      .withTypeAnnotationDescriptor(typeAnnotationDescriptor)
      .withDecorators(decorators)
      .withParameters(parameters)
      .build();
  }
}
