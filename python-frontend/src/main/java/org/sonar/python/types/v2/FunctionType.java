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
package org.sonar.python.types.v2;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * FunctionType
 */
public record FunctionType(
  String name,
  List<PythonType> attributes,
  List<ParameterV2> parameters,
  PythonType returnType,
  boolean isAsynchronous,
  boolean hasDecorators,
  boolean isInstanceMethod,
  boolean hasVariadicParameter,
  @Nullable PythonType owner
) implements PythonType {

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    FunctionType that = (FunctionType) o;
    return hasDecorators == that.hasDecorators
      && isAsynchronous == that.isAsynchronous
      && isInstanceMethod == that.isInstanceMethod
      && hasVariadicParameter == that.hasVariadicParameter
      && Objects.equals(name, that.name)
      && Objects.equals(returnType, that.returnType)
      && Objects.equals(attributes, that.attributes)
      && Objects.equals(parameters, that.parameters);
  }

  @Override
  public Optional<String> displayName() {
    return Optional.of("Callable");
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, attributes, parameters, returnType, isAsynchronous, hasDecorators, isInstanceMethod, hasVariadicParameter);
  }
}
