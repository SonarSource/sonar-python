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

import java.util.stream.Collectors;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class OverloadedFunctionSymbolToDescriptorConverter {

  private final FunctionSymbolToDescriptorConverter functionConverter;

  public OverloadedFunctionSymbolToDescriptorConverter(FunctionSymbolToDescriptorConverter functionConverter) {
    this.functionConverter = functionConverter;
  }

  public AmbiguousDescriptor convert(SymbolsProtos.OverloadedFunctionSymbol overloadedFunctionSymbol) {
    return convert(overloadedFunctionSymbol, false);
  }

  public AmbiguousDescriptor convert(SymbolsProtos.OverloadedFunctionSymbol overloadedFunctionSymbol, boolean isParentIsAClass) {
    if (overloadedFunctionSymbol.getDefinitionsList().size() < 2) {
      throw new IllegalStateException("Overloaded function symbols should have at least two definitions.");
    }
    var name = overloadedFunctionSymbol.getName();
    var fullyQualifiedName = TypeShedUtils.normalizedFqn(overloadedFunctionSymbol.getFullname());
    var descriptors = overloadedFunctionSymbol.getDefinitionsList().stream()
      .map(fs -> functionConverter.convert(fs, isParentIsAClass))
      .map(Descriptor.class::cast)
      .collect(Collectors.toSet());
    return new AmbiguousDescriptor(name, fullyQualifiedName, descriptors);
  }

}
