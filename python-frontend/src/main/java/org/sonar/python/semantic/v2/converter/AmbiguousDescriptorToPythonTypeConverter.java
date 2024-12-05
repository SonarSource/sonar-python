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

import java.util.stream.Collectors;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

public class AmbiguousDescriptorToPythonTypeConverter implements DescriptorToPythonTypeConverter {

  public PythonType convert(ConversionContext ctx, AmbiguousDescriptor from) {
    var candidates = from.alternatives().stream().map(ctx::convert).collect(Collectors.toSet());
    return UnionType.or(candidates);
  }

  @Override
  public PythonType convert(ConversionContext ctx, Descriptor from) {
    if (from instanceof AmbiguousDescriptor ambiguousDescriptor) {
      return convert(ctx, ambiguousDescriptor);
    }
    throw new IllegalArgumentException("Unsupported Descriptor");
  }
}
