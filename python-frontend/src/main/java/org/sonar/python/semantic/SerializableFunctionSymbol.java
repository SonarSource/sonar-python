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

import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.Symbol;

public class SerializableFunctionSymbol extends SerializableSymbol {
  private final List<SerializableParameter> parameters;
  private final boolean isStub;
  private final boolean isInstanceMethod;
  private final List<String> decorators;
  private final LocationInFile definitionLocation;

  public SerializableFunctionSymbol(String name, @Nullable String fullyQualifiedName, List<SerializableParameter> parameters, boolean isStub,
                                    boolean isInstanceMethod, List<String> decorators, LocationInFile definitionLocation
                                        ) {
    super(name, fullyQualifiedName);
    this.parameters = parameters;
    this.isStub = isStub;
    this.isInstanceMethod = isInstanceMethod;
    this.decorators = decorators;
    this.definitionLocation = definitionLocation;
  }

  public List<SerializableParameter> parameters() {
    return parameters;
  }

  public boolean isStub() {
    return isStub;
  }

  public boolean isInstanceMethod() {
    return isInstanceMethod;
  }

  public List<String> decorators() {
    return decorators;
  }

  @CheckForNull
  public LocationInFile definitionLocation() {
    return definitionLocation;
  }

  @Override
  public Symbol toSymbol() {
    return new FunctionSymbolImpl(this);
  }
}
