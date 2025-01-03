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
package org.sonar.python.semantic.v2;

import java.util.HashMap;
import java.util.Map;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeWrapper;

public class LazyTypesContext {
  private final Map<String, LazyType> lazyTypes;
  private final TypeTable typeTable;

  public LazyTypesContext(ProjectLevelTypeTable typeTable) {
    this.lazyTypes = new HashMap<>();
    this.typeTable = typeTable;
  }

  public TypeWrapper getOrCreateLazyTypeWrapper(String importPath) {
    return new LazyTypeWrapper(getOrCreateLazyType(importPath));
  }

  public LazyType getOrCreateLazyType(String importPath) {
    if (lazyTypes.containsKey(importPath)) {
      return lazyTypes.get(importPath);
    }
    var lazyType = new LazyType(importPath, this);
    lazyTypes.put(importPath, lazyType);
    return lazyType;
  }

  public PythonType resolveLazyType(LazyType lazyType) {
    PythonType resolved = typeTable.getType(lazyType.importPath());
    lazyType.resolve(resolved);
    lazyTypes.remove(lazyType.importPath());
    return resolved;
  }
}
