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
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.LocationInFile;

/**
 * FunctionType
 */
@Beta
public final class FunctionType implements PythonType {
  private final String name;
  private final List<PythonType> attributes;
  private final List<ParameterV2> parameters;
  private final PythonType returnType;
  private final boolean isAsynchronous;
  private final boolean hasDecorators;
  private final boolean isInstanceMethod;
  private final boolean hasVariadicParameter;
  @Nullable
  private final PythonType owner;
  @Nullable
  private final LocationInFile locationInFile;

  /**
   *
   */
  public FunctionType(
    String name,
    List<PythonType> attributes,
    List<ParameterV2> parameters,
    PythonType returnType,
    boolean isAsynchronous,
    boolean hasDecorators,
    boolean isInstanceMethod,
    boolean hasVariadicParameter,
    @Nullable PythonType owner,
    @Nullable LocationInFile locationInFile
  ) {
    this.name = name;
    this.attributes = attributes;
    this.parameters = parameters;
    this.returnType = returnType;
    this.isAsynchronous = isAsynchronous;
    this.hasDecorators = hasDecorators;
    this.isInstanceMethod = isInstanceMethod;
    this.hasVariadicParameter = hasVariadicParameter;
    this.owner = owner;
    this.locationInFile = locationInFile;
  }
/*

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
*/

  @Override
  public Optional<String> displayName() {
    return Optional.of("Callable");
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return Optional.ofNullable(this.locationInFile);
  }

/*  @Override
  public int hashCode() {
    return Objects.hash(name, attributes, parameters, returnType, isAsynchronous, hasDecorators, isInstanceMethod, hasVariadicParameter);
  }*/

  @Override
  public String toString() {
    return "FunctionType[%s]".formatted(name);
  }

  @Override
  public String name() {
    return name;
  }

  public List<PythonType> attributes() {
    return attributes;
  }

  public List<ParameterV2> parameters() {
    return parameters;
  }

  public PythonType returnType() {
    return returnType;
  }

  public boolean isAsynchronous() {
    return isAsynchronous;
  }

  public boolean hasDecorators() {
    return hasDecorators;
  }

  public boolean isInstanceMethod() {
    return isInstanceMethod;
  }

  public boolean hasVariadicParameter() {
    return hasVariadicParameter;
  }

  @Nullable
  public PythonType owner() {
    return owner;
  }

  @Nullable
  public LocationInFile locationInFile() {
    return locationInFile;
  }

}
