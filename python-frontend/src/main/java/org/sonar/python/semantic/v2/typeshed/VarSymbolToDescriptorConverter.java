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
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class VarSymbolToDescriptorConverter {

  public Descriptor convert(SymbolsProtos.VarSymbol varSymbol) {
    var typeAnnotation = Optional.of(varSymbol)
      .map(SymbolsProtos.VarSymbol::getTypeAnnotation)
      .map(SymbolsProtos.Type::getFullyQualifiedName)
      .orElse(null);
    return new VariableDescriptor(varSymbol.getName(), varSymbol.getFullyQualifiedName(), typeAnnotation);
  }

}
