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

import java.util.List;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class TypeSymbolToDescriptorConverter {

  TypeAnnotationDescriptor convert(SymbolsProtos.Type type) {
    List<TypeAnnotationDescriptor> args = type.getArgsList().stream()
      .map(this::convert)
      .toList();
    TypeAnnotationDescriptor.TypeKind kind = TypeAnnotationDescriptor.TypeKind.valueOf(type.getKind().name());
    String normalizedFqn = TypeShedUtils.normalizedFqn(type.getFullyQualifiedName());
    return new TypeAnnotationDescriptor(
      type.getPrettyPrintedName(),
      kind,
      args,
      normalizedFqn);
  }
}
