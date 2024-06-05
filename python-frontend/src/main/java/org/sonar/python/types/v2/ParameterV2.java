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

import javax.annotation.Nullable;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.LocationInFile;

@Beta
public class ParameterV2 {
  @Nullable
  private final String name;
  private PythonType declaredType;
  private final boolean hasDefaultValue;
  private final boolean isKeywordOnly;
  private final boolean isPositionalOnly;
  private final boolean isKeywordVariadic;
  private final boolean isPositionalVariadic;
  @Nullable
  private final LocationInFile location;

  public ParameterV2(
    @Nullable String name,
    PythonType declaredType,
    boolean hasDefaultValue,
    boolean isKeywordOnly,
    boolean isPositionalOnly,
    boolean isKeywordVariadic,
    boolean isPositionalVariadic,
    @Nullable LocationInFile location) {
    this.name = name;
    this.declaredType = declaredType;
    this.hasDefaultValue = hasDefaultValue;
    this.isKeywordOnly = isKeywordOnly;
    this.isPositionalOnly = isPositionalOnly;
    this.isKeywordVariadic = isKeywordVariadic;
    this.isPositionalVariadic = isPositionalVariadic;
    this.location = location;
  }

  @Beta
  public boolean isVariadic() {
    return isKeywordVariadic || isPositionalVariadic;
  }

  @Nullable
  public String name() {
    return name;
  }

  public void declaredType(PythonType declaredType) {
    this.declaredType = declaredType;
  }

  public PythonType declaredType() {
    return declaredType;
  }

  public boolean hasDefaultValue() {
    return hasDefaultValue;
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

  @Nullable
  public LocationInFile location() {
    return location;
  }

}

