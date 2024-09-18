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
package org.sonar.python.semantic.v2.typeshed;

import java.util.Optional;
import java.util.function.Predicate;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class ParameterSymbolToDescriptorConverter {

  FunctionDescriptor.Parameter convert(SymbolsProtos.ParameterSymbol parameter) {
    var annotatedType = Optional.of(parameter)
      .map(SymbolsProtos.ParameterSymbol::getTypeAnnotation)
      .map(SymbolsProtos.Type::getFullyQualifiedName)
      .filter(Predicate.not(String::isEmpty))
      .orElse(null);
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
