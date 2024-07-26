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
import java.util.Optional;
import javax.annotation.CheckForNull;
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
  private PythonType returnType;
  private final boolean isAsynchronous;
  private final boolean hasDecorators;
  private final boolean isInstanceMethod;
  private final boolean hasVariadicParameter;
  private final PythonType owner;
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

  @Override
  public Optional<String> displayName() {
    return Optional.of("Callable");
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return Optional.ofNullable(this.locationInFile);
  }

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
    if (returnType instanceof LazyType lazyType) {
      return lazyType.resolve();
    }
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

  @CheckForNull
  public PythonType owner() {
    return owner;
  }

  public void resolveLazyReturnType(PythonType pythonType) {
    if (!(returnType instanceof LazyType)) {
      throw new IllegalStateException("Trying to resolve an already resolved lazy type.");
    }
    this.returnType = pythonType;
  }
}
