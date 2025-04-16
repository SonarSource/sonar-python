/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.types.v2;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TriBool;

public class TypeCheckMap<V> {
  private final Map<TypeCheckBuilder, V> map;

  public TypeCheckMap() {
    this(new HashMap<>());
  }

  public TypeCheckMap(Map<TypeCheckBuilder, V> map) {
    this.map = map;
  }

  @SafeVarargs
  public static <V> TypeCheckMap<V> ofEntries(Map.Entry<TypeCheckBuilder, V>... entries) {
    var typeCheckMap = new TypeCheckMap<V>();
    for (var entry : entries) {
      typeCheckMap.put(entry.getKey(), entry.getValue());
    }
    return typeCheckMap;
  }

  public V put(TypeCheckBuilder key, V value) {
    return map.put(key, value);
  }

  @CheckForNull
  public V getForType(PythonType type) {
    return getOptionalForType(type).orElse(null);
  }

  public Optional<V> getOptionalForType(PythonType type) {
    return map.entrySet()
      .stream()
      .filter(entry -> entry.getKey().check(type) == TriBool.TRUE)
      .findFirst()
      .map(Map.Entry::getValue);
  }
}
