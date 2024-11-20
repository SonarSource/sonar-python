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

import javax.annotation.Nullable;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class VarSymbolToDescriptorConverter {

  public Descriptor convert(SymbolsProtos.VarSymbol varSymbol) {
    var fullyQualifiedName = TypeShedUtils.normalizedFqn(varSymbol.getFullyQualifiedName());
    SymbolsProtos.Type protoTypeAnnotation = varSymbol.getTypeAnnotation();
    var isImportedModule = varSymbol.getIsImportedModule();
    var typeAnnotation = TypeShedUtils.getTypesNormalizedFqn(protoTypeAnnotation);
    if (isTypeAnnotationKnownToBeIncorrect(fullyQualifiedName)) {
      return new VariableDescriptor(varSymbol.getName(), fullyQualifiedName, null, isImportedModule);
    }
    return new VariableDescriptor(varSymbol.getName(), fullyQualifiedName, typeAnnotation, isImportedModule);
  }

  private static boolean isTypeAnnotationKnownToBeIncorrect(@Nullable String fullyQualifiedName) {
    // TypedDict is defined to have type "object" in Typeshed, which is incorrect and leads to FPs
    return "typing.TypedDict".equals(fullyQualifiedName) || "typing_extensions.TypedDict".equals(fullyQualifiedName);
  }

}
