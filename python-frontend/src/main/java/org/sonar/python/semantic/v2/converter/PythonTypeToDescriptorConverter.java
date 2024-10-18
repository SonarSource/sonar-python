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

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;
import org.sonar.python.types.v2.UnknownType;

public class PythonTypeToDescriptorConverter {

  public Descriptor convert(String moduleFqn, SymbolV2 symbol, Set<PythonType> types) {
    var candidates = types.stream()
      .map(type -> convert(moduleFqn, symbol, type))
      .collect(Collectors.toSet());

    if (candidates.size() == 1) {
      return candidates.iterator().next();
    }
    return new AmbiguousDescriptor(symbol.name(), symbolFqn(moduleFqn, symbol), candidates);
  }

  private Descriptor convert(String moduleFqn, SymbolV2 symbol, PythonType type) {
    if (type instanceof FunctionType functionType) {
      return convert(moduleFqn, symbol, functionType);
    }
    if (type instanceof ClassType classType) {
      return convert(moduleFqn, symbol, classType);
    }
    if (type instanceof UnionType unionType) {
      return convert(moduleFqn, symbol, unionType);
    }
    if (type instanceof UnknownType.UnresolvedImportType unresolvedImportType) {
      return convert(moduleFqn, symbol, unresolvedImportType);
    }
    return new VariableDescriptor(symbol.name(), symbolFqn(moduleFqn, symbol), null);
  }

  private Descriptor convert(String moduleFqn, SymbolV2 symbol, FunctionType type) {

    var parameters = type.parameters()
      .stream()
      .map(parameter -> convert(parameter))
      .toList();

    return new FunctionDescriptor(symbol.name(), symbolFqn(moduleFqn, symbol),
      parameters,
      type.isAsynchronous(),
      type.isInstanceMethod(),
      List.of(),
      type.hasDecorators(),
      type.definitionLocation().orElse(null),
      null,
      null
      );
  }

  private Descriptor convert(String moduleFqn, SymbolV2 symbol, ClassType type) {
    return new ClassDescriptor(symbol.name(), symbolFqn(moduleFqn, symbol),
      List.of(),
      Set.of(),
      type.hasDecorators(),
      type.definitionLocation().orElse(null),
      false,
      type.hasMetaClass(),
      null,
      false
    );
  }

  private Descriptor convert(String moduleFqn, SymbolV2 symbol, UnionType type) {
    var candidates = type.candidates().stream()
      .map(candidateType -> convert(moduleFqn, symbol, candidateType))
      .collect(Collectors.toSet());
    return new AmbiguousDescriptor(symbol.name(),
      symbolFqn(moduleFqn, symbol),
      candidates
    );
  }

  private Descriptor convert(String moduleFqn, SymbolV2 symbol, UnknownType.UnresolvedImportType type) {
    return new VariableDescriptor(symbol.name(),
      symbolFqn(moduleFqn, symbol),
      type.importPath()
    );
  }

  public FunctionDescriptor.Parameter convert(ParameterV2 parameter) {
    String annotatedType = null;
    var type = parameter.declaredType().type();
    if (type instanceof UnknownType.UnresolvedImportType importType) {
      annotatedType = importType.importPath();
    }

    return new FunctionDescriptor.Parameter(parameter.name(),
      annotatedType,
      parameter.hasDefaultValue(),
      parameter.isKeywordOnly(),
      parameter.isPositionalOnly(),
      parameter.isKeywordVariadic(),
      parameter.isPositionalVariadic(),
      parameter.location());
  }

  private static String symbolFqn(String moduleFqn, SymbolV2 symbol) {
    return moduleFqn + "." + symbol.name();
  }


}
