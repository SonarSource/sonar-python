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
package org.sonar.python.semantic.v2;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.LazyTypeWrapper;

public class LazyTypesContext {
  private final Map<String, LazyType> lazyTypes;
  private final TypeTable typeTable;

  public LazyTypesContext(ProjectLevelTypeTable typeTable) {
    this.lazyTypes = new ConcurrentHashMap<>();
    this.typeTable = typeTable;
  }

  public TypeWrapper getOrCreateLazyTypeWrapper(String importPath) {
    return new LazyTypeWrapper(getOrCreateLazyType(importPath));
  }

  public LazyType getOrCreateLazyType(String importPath) {
    return lazyTypes.computeIfAbsent(importPath, ip -> new LazyType(ip, this));
  }

  public PythonType resolveLazyType(LazyType lazyType) {
    PythonType resolved = typeTable.getType(lazyType.importPath());
    lazyType.resolve(resolved);
    lazyTypes.remove(lazyType.importPath());
    return resolved;
  }
}
