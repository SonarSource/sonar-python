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

import org.sonar.python.index.AliasDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.types.v2.PythonType;

public class AliasDescriptorToPythonTypeConverter implements DescriptorToPythonTypeConverter {
  @Override
  public PythonType convert(ConversionContext ctx, Descriptor from) {
    // We should try to retrieve the original type if possible, instead of recreating it
    return ctx.convert(((AliasDescriptor) from).originalDescriptor());
  }
}