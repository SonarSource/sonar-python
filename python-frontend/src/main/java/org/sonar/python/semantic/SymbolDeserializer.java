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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.types.TypeShed;

public class SymbolDeserializer {

  private final Map<String, Symbol> deserializedSymbolsByFqn = new HashMap<>();
  private final ProjectLevelSymbolTable projectLevelSymbolTable;

  public SymbolDeserializer(ProjectLevelSymbolTable projectLevelSymbolTable) {
    this.projectLevelSymbolTable = projectLevelSymbolTable;
  }

  @CheckForNull
  Symbol deserializeSymbol(@Nullable Set<SerializableSymbol> serializableSymbols) {
    if (serializableSymbols == null || serializableSymbols.isEmpty()) {
      return null;
    }
    String fullyQualifiedName = serializableSymbols.iterator().next().fullyQualifiedName();
//    Symbol deserializedSymbol = deserializedSymbolsByFqn.get(fullyQualifiedName);
//    if (deserializedSymbol != null) {
//      return deserializedSymbol;
//    }
    Set<Symbol> deserializedSymbols = serializableSymbols.stream().map(this::deserialize).collect(Collectors.toSet());
    Symbol deserializedSymbol;
    if (deserializedSymbols.size() > 1) {
      deserializedSymbol = AmbiguousSymbolImpl.create(deserializedSymbols);
    } else {
      deserializedSymbol = deserializedSymbols.iterator().next();
    }
//    deserializedSymbolsByFqn.put(deserializedSymbol.fullyQualifiedName(), deserializedSymbol);
    return deserializedSymbol;
  }

  private Symbol deserialize(SerializableSymbol serializableSymbol) {
    Symbol deserializedSymbol;
    if (serializableSymbol instanceof SerializableClassSymbol) {
      deserializedSymbol = deserializeClass((SerializableClassSymbol) serializableSymbol);
    } else {
      deserializedSymbol = serializableSymbol.toSymbol();
    }
    return deserializedSymbol;
  }

  private ClassSymbol deserializeClass(SerializableClassSymbol serializableSymbol) {
    ClassSymbolImpl classSymbol = (ClassSymbolImpl) serializableSymbol.toSymbol();
    serializableSymbol.superClasses().stream()
      .map(fqn -> fqn.equals(classSymbol.fullyQualifiedName) ? classSymbol : resolveSuperClassSymbol(fqn))
      .forEach(superClass -> {
        if (superClass != null) {
          classSymbol.addSuperClass(superClass);
        } else {
          classSymbol.setHasSuperClassWithoutSymbol();
        }
      });
    Set<Symbol> members = serializableSymbol.declaredMembers().stream()
      .map(member -> deserializeSymbol(Collections.singleton(member)))
      .filter(Objects::nonNull)
      .collect(Collectors.toSet());
    classSymbol.addMembers(members);
    return classSymbol;
  }

  @CheckForNull
  Set<Symbol> deserializeSymbols(@Nullable Set<SerializableSymbol> serializableSymbols) {
    if (serializableSymbols == null) {
      return null;
    }
    Map<String, Set<SerializableSymbol>> serializedSymbolsWithSameName = serializableSymbols.stream()
      .collect(Collectors.groupingBy(SerializableSymbol::name, Collectors.toSet()));
    return serializedSymbolsWithSameName.values().stream()
      .map(this::deserializeSymbol)
      .collect(Collectors.toSet());
  }

  private Symbol resolveSuperClassSymbol(String fullyQualifiedName) {
    Symbol symbol = deserializeSymbol(projectLevelSymbolTable.getSymbol(fullyQualifiedName));
    if (symbol == null) {
      symbol = TypeShed.symbolWithFQN(fullyQualifiedName);
      if (symbol != null) {
        symbol = ((SymbolImpl) symbol).copyWithoutUsages();
      }
    }
    if (symbol == null) {
      String[] names = fullyQualifiedName.split("\\.");
      symbol = new SymbolImpl(names[names.length - 1], fullyQualifiedName);
    }
    return symbol;
  }
}
