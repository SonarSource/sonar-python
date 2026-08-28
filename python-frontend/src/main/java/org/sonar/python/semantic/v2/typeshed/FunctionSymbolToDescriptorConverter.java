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
package org.sonar.python.semantic.v2.typeshed;

import java.util.Collection;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;
import org.sonar.python.types.TypeShedTypeTable;

public class FunctionSymbolToDescriptorConverter {

  private final ParameterSymbolToDescriptorConverter parameterConverter;
  private final TypeSymbolToDescriptorConverter typeConverter;
  private final TypeShedTypeTable typeTable;

  public FunctionSymbolToDescriptorConverter() {
    this(TypeShedTypeTable.EMPTY);
  }

  public FunctionSymbolToDescriptorConverter(TypeShedTypeTable typeTable) {
    this.typeTable = typeTable;
    parameterConverter = new ParameterSymbolToDescriptorConverter(typeTable);
    typeConverter = new TypeSymbolToDescriptorConverter(typeTable);
  }

  public FunctionDescriptor convert(SymbolsProtos.FunctionSymbol functionSymbol) {
    return convert(functionSymbol, false, null);
  }

  public FunctionDescriptor convert(SymbolsProtos.FunctionSymbol functionSymbol, boolean isParentIsAClass) {
    return convert(functionSymbol, isParentIsAClass, null);
  }

  public FunctionDescriptor convert(SymbolsProtos.FunctionSymbol functionSymbol, boolean isParentIsAClass, @Nullable String containerFqn) {
    var fullyQualifiedName = TypeShedUtils.normalizedSymbolFqn(
      functionSymbol.getFullyQualifiedName(), containerFqn, functionSymbol.getName());
    TypeAnnotationDescriptor typeAnnotationDescriptor = null;
    SymbolsProtos.Type returnAnnotation = typeTable.resolve(
      functionSymbol.getReturnAnnotationId(), functionSymbol.hasReturnAnnotation(), functionSymbol.getReturnAnnotation());
    if (returnAnnotation != null) {
      typeAnnotationDescriptor = typeConverter.convert(returnAnnotation);
    }
    String returnType = TypeShedUtils.getTypesNormalizedFqn(returnAnnotation, typeTable);
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
    var isClassMethod = isParentIsAClass && !functionSymbol.getIsStatic() && functionSymbol.getIsClassMethod();
    return new FunctionDescriptor.FunctionDescriptorBuilder()
      .withName(functionSymbol.getName())
      .withFullyQualifiedName(fullyQualifiedName)
      .withIsAsynchronous(functionSymbol.getIsAsynchronous())
      .withIsInstanceMethod(isInstanceMethod)
      .withIsClassMethod(isClassMethod)
      .withHasDecorators(functionSymbol.getHasDecorators())
      .withAnnotatedReturnTypeName(returnType)
      .withTypeAnnotationDescriptor(typeAnnotationDescriptor)
      .withDecorators(decorators)
      .withParameters(parameters)
      .build();
  }
}
