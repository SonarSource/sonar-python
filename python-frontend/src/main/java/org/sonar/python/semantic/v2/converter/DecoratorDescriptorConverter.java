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
package org.sonar.python.semantic.v2.converter;

import java.util.Comparator;
import java.util.List;
import org.sonar.plugins.python.api.types.v2.DecoratorInfo;
import org.sonar.plugins.python.api.types.v2.FullyQualifiedNameHelper;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.index.DecoratorDescriptor;
import org.sonar.python.index.DecoratorDescriptor.TypeReference;
import org.sonar.python.types.v2.LazyUnionType;

public final class DecoratorDescriptorConverter {
  private DecoratorDescriptorConverter() {
  }

  public static DecoratorDescriptor toDescriptor(DecoratorInfo decorator) {
    return new DecoratorDescriptor(references(decorator.expressionType().type()),
      decorator.calleeType().map(t -> references(t.type())));
  }

  private static List<TypeReference> references(PythonType type) {
    return references(type, false);
  }

  private static List<TypeReference> references(PythonType type, boolean instance) {
    if (type instanceof ObjectType objectType) {
      return references(objectType.type(), true);
    }
    if (type instanceof UnionType union) {
      return union.candidates().stream().flatMap(candidate -> references(candidate, instance).stream()).distinct()
        .sorted(Comparator.comparing(TypeReference::fullyQualifiedName, Comparator.nullsFirst(Comparator.naturalOrder()))
          .thenComparing(TypeReference::isInstance))
        .toList();
    }
    // BasicTypeTable deliberately defers imports during indexing; preserve their
    // paths for lazy resolution.
    if (type instanceof UnknownType.UnresolvedImportType(var importPath)) {
      return List.of(new TypeReference(importPath, instance));
    }
    // Accepted identity loss on redefinition: CLP-1206
    String name = type instanceof UnknownType ? null
      : FullyQualifiedNameHelper.getFullyQualifiedName(type).orElse(null);
    return List.of(name == null ? TypeReference.UNKNOWN : new TypeReference(name, instance));
  }

  public static DecoratorInfo toType(ConversionContext context, DecoratorDescriptor decorator) {
    return new DecoratorInfo(type(context, decorator.expressionTypes()),
      decorator.calleeTypes().map(refs -> type(context, refs)).orElse(null));
  }

  private static TypeWrapper type(ConversionContext context, List<TypeReference> references) {
    var types = references.stream().map(reference -> {
      String name = reference.fullyQualifiedName();
      if (name == null) {
        return PythonType.UNKNOWN;
      }
      PythonType value = context.lazyTypesContext().getOrCreateLazyType(name);
      return reference.isInstance() ? ObjectType.fromType(value) : value;
    }).toList();
    return TypeWrapper.of(LazyUnionType.or(types));
  }
}
