/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.semantic.v2.typetable;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.sonar.plugins.python.api.types.v2.PythonType;

public class CachedTypeTable implements TypeTable {

  private final TypeTable delegate;
  private final Map<List<String>, PythonType> typeCache;
  private final Map<List<String>, PythonType> moduleTypeCache;

  public CachedTypeTable(TypeTable delegate) {
    this.delegate = delegate;
    this.typeCache = new ConcurrentHashMap<>();
    this.moduleTypeCache = new ConcurrentHashMap<>();
  }

  @Override
  public PythonType getBuiltinsModule() {
    return delegate.getBuiltinsModule();
  }

  @Override
  public PythonType getType(String typeFqn) {
    List<String> parts = List.of(typeFqn.split("\\."));
    return getType(parts);
  }

  @Override
  public PythonType getType(String... typeFqnParts) {
    return getType(List.of(typeFqnParts));
  }

  @Override
  public PythonType getType(List<String> typeFqnParts) {
    PythonType cachedResult = typeCache.get(typeFqnParts);
    if (cachedResult != null) {
      return cachedResult;
    }

    PythonType computedResult = delegate.getType(typeFqnParts);
    typeCache.putIfAbsent(typeFqnParts, computedResult);
    return computedResult;
  }

  @Override
  public PythonType getModuleType(List<String> typeFqnParts) {
    PythonType cachedResult = moduleTypeCache.get(typeFqnParts);
    if (cachedResult != null) {
      return cachedResult;
    }

    PythonType computedResult = delegate.getModuleType(typeFqnParts);
    moduleTypeCache.putIfAbsent(typeFqnParts, computedResult);
    return computedResult;
  }
}
