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

import java.util.Optional;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.SimpleTypeWrapper;
import org.sonar.python.types.v2.TypeWrapper;

public class ParameterConverter {

  public ParameterV2 convert(ConversionContext ctx, FunctionDescriptor.Parameter parameter) {
    var typeWrapper = Optional.ofNullable(parameter.annotatedType())
      .map(fqn -> (PythonType) ctx.lazyTypesContext().getOrCreateLazyType(fqn))
      .map(lt -> (TypeWrapper) new LazyTypeWrapper(lt))
      .orElseGet(() -> new SimpleTypeWrapper(PythonType.UNKNOWN));

    var type = new ObjectType(typeWrapper);

    return new ParameterV2(parameter.name(),
      new SimpleTypeWrapper(type),
      parameter.hasDefaultValue(),
      parameter.isKeywordOnly(),
      parameter.isPositionalOnly(),
      parameter.isKeywordVariadic(),
      parameter.isPositionalVariadic(),
      parameter.location());
  }

}
