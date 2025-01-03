/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
