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

import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class ParameterSymbolToDescriptorConverter {

  FunctionDescriptor.Parameter convert(SymbolsProtos.ParameterSymbol parameter) {
    var annotatedType = TypeShedUtils.getTypesNormalizedFqn(parameter.getTypeAnnotation());
    var isKeywordOnly = parameter.getKind() == SymbolsProtos.ParameterKind.KEYWORD_ONLY;
    var isPositionalOnly = parameter.getKind() == SymbolsProtos.ParameterKind.POSITIONAL_ONLY;
    var isPositionalVariadic = parameter.getKind() == SymbolsProtos.ParameterKind.VAR_POSITIONAL;
    var isKeywordVariadic = parameter.getKind() == SymbolsProtos.ParameterKind.VAR_KEYWORD;

    return new FunctionDescriptor.Parameter(
      parameter.getName(),
      annotatedType,
      parameter.getHasDefault(),
      isKeywordOnly,
      isPositionalOnly,
      isPositionalVariadic,
      isKeywordVariadic,
      null);
  }

}
