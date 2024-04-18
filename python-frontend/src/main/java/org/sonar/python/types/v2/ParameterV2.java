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

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;

public class ParameterV2 {

  private final String name;
  private PythonType declaredType;
  private final boolean hasDefaultValue;
  private final boolean isKeywordVariadic;
  private final boolean isPositionalVariadic;
  private final boolean isKeywordOnly;
  private final boolean isPositionalOnly;
  private final LocationInFile location;

  public ParameterV2(@Nullable String name, PythonType declaredType, boolean hasDefaultValue,
    FunctionTypeBuilder.ParameterState parameterState, boolean isKeywordVariadic, boolean isPositionalVariadic,
    @Nullable LocationInFile location) {
    this.name = name;
    this.declaredType = declaredType;
    this.hasDefaultValue = hasDefaultValue;
    this.isKeywordVariadic = isKeywordVariadic;
    this.isPositionalVariadic = isPositionalVariadic;
    this.isKeywordOnly = parameterState.keywordOnly;
    this.isPositionalOnly = parameterState.positionalOnly;
    this.location = location;
  }

  @CheckForNull
  public String name() {
    return name;
  }

  public PythonType declaredType() {
    return declaredType;
  }

  public boolean hasDefaultValue() {
    return hasDefaultValue;
  }

  public boolean isVariadic() {
    return isKeywordVariadic || isPositionalVariadic;
  }

  public boolean isKeywordOnly() {
    return isKeywordOnly;
  }

  public boolean isPositionalOnly() {
    return isPositionalOnly;
  }

  public boolean isKeywordVariadic() {
    return isKeywordVariadic;
  }

  public boolean isPositionalVariadic() {
    return isPositionalVariadic;
  }

  @CheckForNull
  public LocationInFile location() {
    return location;
  }
}

