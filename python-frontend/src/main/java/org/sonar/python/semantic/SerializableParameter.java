/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.semantic;

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;

public class SerializableParameter {
  @Nullable
  private final String name;
  private final String declaredType;
  private final boolean hasDefaultValue;
  private final boolean isKeywordOnly;
  private final boolean isPositionalOnly;
  private final boolean isVariadic;
  @Nullable
  private final LocationInFile location;


  public SerializableParameter(FunctionSymbol.Parameter parameter) {
    name = parameter.name();
    declaredType = parameter.declaredType().toString();
    hasDefaultValue = parameter.hasDefaultValue();
    isKeywordOnly = parameter.isKeywordOnly();
    isPositionalOnly = parameter.isPositionalOnly();
    isVariadic = parameter.isVariadic();
    location = parameter.location();
  }

  @CheckForNull
  public String name() {
    return name;
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

  public boolean isVariadic() {
    return isVariadic;
  }

  @CheckForNull
  public LocationInFile location() {
    return location;
  }
}
