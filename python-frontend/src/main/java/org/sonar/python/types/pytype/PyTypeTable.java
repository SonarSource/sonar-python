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
package org.sonar.python.types.pytype;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.tree.Name;

public class PyTypeTable {

  private final Map<String, List<PyTypeInfo>> files;
  private final Map<TypePositionKey, PyTypeInfo> typesByPosition;
  private final HashMap<TypePositionKey, List<PyTypeInfo>> multipleTypesByPosition;

  public PyTypeTable(Map<String, List<PyTypeInfo>> files) {
    this.files = files;

    typesByPosition = files.entrySet()
      .stream()
      .flatMap(entry -> {
        var file = entry.getKey();
        return entry.getValue()
          .stream()
          .map(typeInfo -> Map.entry(
            new TypePositionKey(file, typeInfo.startLine(), typeInfo.startCol(), typeInfo.text()),
            typeInfo));
      })
      .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (v1, v2) -> v1));

    multipleTypesByPosition = files
      .entrySet()
      .stream()
      .flatMap(entry -> {
        var file = entry.getKey();
        return entry.getValue()
          .stream()
          .map(typeInfo -> Map.entry(
            new TypePositionKey(file, typeInfo.startLine(), typeInfo.startCol(), typeInfo.text()),
            typeInfo));
      })
      .collect(Collectors.groupingBy(Map.Entry::getKey, HashMap::new, Collectors.mapping(Map.Entry::getValue, Collectors.toList())));
  }

  public Optional<PyTypeInfo> getVariableTypeFor(String fileName, Name name) {
    var token = name.firstToken();
    return getVariableTypeFor(fileName, token.line(), token.column(), name.name(), "Variable");
  }

  public Optional<PyTypeInfo> getFunctionTypeFor(String fileName, Name name) {
    var token = name.parent().firstToken();
    return getVariableTypeFor(fileName, token.line(), token.column(), name.name(), "Function");
  }

  public Optional<PyTypeInfo> getVariableTypeFor(String fileName, int line, int column, String name, String kind) {
    TypePositionKey typePositionKey = new TypePositionKey(fileName, line, column, name);
    return Optional.ofNullable(multipleTypesByPosition.get(typePositionKey))
      .filter(Predicate.not(List::isEmpty))
      .map(types -> types.stream()
        .filter(t -> kind.equals(t.syntaxRole()))
        .findFirst()
        .orElse(types.get(0)));
  }

  private record TypePositionKey(String fileName, int line, int column, String name) {
  }


}
