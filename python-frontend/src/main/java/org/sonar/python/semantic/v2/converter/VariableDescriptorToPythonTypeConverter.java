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
package org.sonar.python.semantic.v2.converter;

import org.sonar.python.index.Descriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.SpecialFormType;
import org.sonar.python.types.v2.TypeWrapper;

public class VariableDescriptorToPythonTypeConverter implements DescriptorToPythonTypeConverter {

  public PythonType convert(ConversionContext ctx, VariableDescriptor from) {
    String fullyQualifiedName = from.fullyQualifiedName();
    if (from.isImportedModule() && fullyQualifiedName != null) {
      return ctx.lazyTypesContext().getOrCreateLazyType(fullyQualifiedName);
    }
    String annotatedType = from.annotatedType();
    if (annotatedType != null) {
      if ("typing._SpecialForm".equals(annotatedType) && fullyQualifiedName != null) {
        // Defensive null check on fullyQualifiedName: it should never be null for SpecialForm
        return new SpecialFormType(fullyQualifiedName);
      }
      TypeWrapper typeWrapper = ctx.lazyTypesContext().getOrCreateLazyTypeWrapper(annotatedType);
      return new ObjectType(typeWrapper);
    }
    return PythonType.UNKNOWN;
  }

  @Override
  public PythonType convert(ConversionContext ctx, Descriptor from) {
    if (from instanceof VariableDescriptor variableDescriptor) {
      return convert(ctx, variableDescriptor);
    }
    throw new IllegalArgumentException("Unsupported Descriptor");
  }
}
