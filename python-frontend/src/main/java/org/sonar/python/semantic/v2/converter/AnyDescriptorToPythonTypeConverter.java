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
package org.sonar.python.semantic.v2.converter;

import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeOrigin;
import org.sonar.python.types.v2.TypeWrapper;

public class AnyDescriptorToPythonTypeConverter {

  private static final DescriptorToPythonTypeConverter UNKNOWN_DESCRIPTOR_CONVERTER = new UnknownDescriptorToPythonTypeConverter();
  private final Map<Descriptor.Kind, DescriptorToPythonTypeConverter> converters;
  private final LazyTypesContext lazyTypesContext;

  public AnyDescriptorToPythonTypeConverter(LazyTypesContext lazyTypesContext) {
    this.lazyTypesContext = lazyTypesContext;
    converters = new EnumMap<>(Map.of(
      Descriptor.Kind.CLASS, new ClassDescriptorToPythonTypeConverter(),
      Descriptor.Kind.FUNCTION, new FunctionDescriptorToPythonTypeConverter(),
      Descriptor.Kind.VARIABLE, new VariableDescriptorToPythonTypeConverter(),
      Descriptor.Kind.AMBIGUOUS, new AmbiguousDescriptorToPythonTypeConverter(),
      Descriptor.Kind.ALIAS, new AliasDescriptorToPythonTypeConverter()
    ));

  }

  public Map<String, TypeWrapper> convertModuleType(String moduleFqn, Map<String, Descriptor> stringDescriptorMap) {
    Map<String, TypeWrapper> moduleMembers = new HashMap<>();
    for (var entry : stringDescriptorMap.entrySet()) {
      var descriptor = entry.getValue();
      var name = entry.getKey();
      PythonType result;
      String fullyQualifiedName = descriptor.fullyQualifiedName();
      String reconstructedFqn = moduleFqn + "." + descriptor.name();
      if (!reconstructedFqn.equals(fullyQualifiedName) && fullyQualifiedName != null) {
        // We create lazy types for descriptors that are not local to the module
        result = lazyTypesContext.getOrCreateLazyType(fullyQualifiedName);
      } else {
        result = this.convert(moduleFqn, descriptor, TypeOrigin.STUB);
      }
      moduleMembers.put(name, TypeWrapper.of(result));
    }
    return moduleMembers;
  }

  public PythonType convert(String moduleFqn, Descriptor from, TypeOrigin typeOrigin) {
    var ctx = new ConversionContext(moduleFqn, lazyTypesContext, this::convert, typeOrigin);
    return convert(ctx, from);
  }

  private PythonType convert(ConversionContext ctx, Descriptor from) {
    return converterFor(from).convert(ctx, from);
  }

  private DescriptorToPythonTypeConverter converterFor(Descriptor descriptor) {
    return converters.getOrDefault(descriptor.kind(), UNKNOWN_DESCRIPTOR_CONVERTER);
  }

}
