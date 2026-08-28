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

import java.util.List;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;
import org.sonar.python.types.TypeShedTypeTable;

public class TypeSymbolToDescriptorConverter {

  private final TypeShedTypeTable typeTable;

  TypeSymbolToDescriptorConverter() {
    this(TypeShedTypeTable.EMPTY);
  }

  TypeSymbolToDescriptorConverter(TypeShedTypeTable typeTable) {
    this.typeTable = typeTable;
  }

  TypeAnnotationDescriptor convert(SymbolsProtos.Type type) {
    List<TypeAnnotationDescriptor> args = typeTable.arguments(type).stream()
      .map(this::convert)
      .toList();
    TypeAnnotationDescriptor.TypeKind kind = TypeAnnotationDescriptor.TypeKind.valueOf(type.getKind().name());
    String normalizedFqn = TypeShedUtils.normalizedFqn(type.getFullyQualifiedName());
    return new TypeAnnotationDescriptor(
      type.getPrettyPrintedName(),
      kind,
      args,
      normalizedFqn,
      type.getIsSelf());
  }
}
