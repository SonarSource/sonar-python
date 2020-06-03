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
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;

public class SerializableClassSymbol extends SerializableSymbol {
  private final List<String> superClasses;
  private final Set<SerializableSymbol> declaredMembers;
  private final boolean hasSuperClassWithoutSymbol;

  public SerializableClassSymbol(String name, @Nullable String fullyQualifiedName,
                                 List<String> superClasses, Set<SerializableSymbol> declaredMembers, boolean hasSuperClassWithoutSymbol) {
    super(name, fullyQualifiedName);
    this.superClasses = superClasses;
    this.declaredMembers = declaredMembers;
    this.hasSuperClassWithoutSymbol = hasSuperClassWithoutSymbol;
  }

  public List<String> superClasses() {
    return superClasses;
  }

  public Set<SerializableSymbol> declaredMembers() {
    return declaredMembers;
  }

  @Override
  public Symbol toSymbol() {
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(name(), fullyQualifiedName());
    if (hasSuperClassWithoutSymbol) {
      classSymbol.setHasSuperClassWithoutSymbol();
    }
    return classSymbol;
  }
}
