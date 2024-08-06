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
package org.sonar.python.semantic.v2;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.PythonType;

public class LazyTypesContext {
  private final Map<String, LazyType> lazyTypes;
  private final ProjectLevelTypeTable projectLevelTypeTable;

  public LazyTypesContext(ProjectLevelTypeTable projectLevelTypeTable) {
    this.lazyTypes = new HashMap<>();
    this.projectLevelTypeTable = projectLevelTypeTable;
  }

  public LazyType getOrCreateLazyType(String fullyQualifiedName) {
    if (lazyTypes.containsKey(fullyQualifiedName)) {
      return lazyTypes.get(fullyQualifiedName);
    }
    var lazyType = new LazyType(fullyQualifiedName, this);
    lazyTypes.put(fullyQualifiedName, lazyType);
    return lazyType;
  }

  public PythonType resolveLazyType(LazyType lazyType) {
    PythonType resolved = projectLevelTypeTable.getType(lazyType.fullyQualifiedName());
    lazyType.resolve(resolved);
    lazyTypes.remove(lazyType.fullyQualifiedName());
    return resolved;
  }
}
