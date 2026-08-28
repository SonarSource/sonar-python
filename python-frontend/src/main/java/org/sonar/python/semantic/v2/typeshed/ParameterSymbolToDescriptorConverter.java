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

import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;
import org.sonar.python.types.TypeShedTypeTable;

public class ParameterSymbolToDescriptorConverter {

  private final TypeShedTypeTable typeTable;
  private final TypeSymbolToDescriptorConverter typeConverter;

  ParameterSymbolToDescriptorConverter() {
    this(TypeShedTypeTable.EMPTY);
  }

  ParameterSymbolToDescriptorConverter(TypeShedTypeTable typeTable) {
    this.typeTable = typeTable;
    this.typeConverter = new TypeSymbolToDescriptorConverter(typeTable);
  }

  FunctionDescriptor.Parameter convert(SymbolsProtos.ParameterSymbol parameter) {
    var type = typeTable.resolve(parameter.getTypeAnnotationId(), parameter.hasTypeAnnotation(), parameter.getTypeAnnotation());
    var annotatedType = TypeShedUtils.getTypesNormalizedFqn(type, typeTable);
    var typeAnnotationDescriptor = type == null ? null : typeConverter.convert(type);
    var isKeywordOnly = parameter.getKind() == SymbolsProtos.ParameterKind.KEYWORD_ONLY;
    var isPositionalOnly = parameter.getKind() == SymbolsProtos.ParameterKind.POSITIONAL_ONLY;
    var isPositionalVariadic = parameter.getKind() == SymbolsProtos.ParameterKind.VAR_POSITIONAL;
    var isKeywordVariadic = parameter.getKind() == SymbolsProtos.ParameterKind.VAR_KEYWORD;

    return new FunctionDescriptor.Parameter(
      parameter.getName(),
      annotatedType,
      typeAnnotationDescriptor,
      parameter.getHasDefault(),
      isKeywordOnly,
      isPositionalOnly,
      isPositionalVariadic,
      isKeywordVariadic,
      null);
  }

}
