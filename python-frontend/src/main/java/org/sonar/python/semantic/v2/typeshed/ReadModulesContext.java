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
package org.sonar.python.semantic.v2.typeshed;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.sonar.python.types.v2.ModuleType;

public class ReadModulesContext {
  private final Set<String> modulesToRead;
  private final Map<String, ModuleType> resolvedModules;
  private final Map<String, PromiseType> promiseTypes;

  public ReadModulesContext(Map<String, ModuleType> resolvedModules) {
    this.resolvedModules = resolvedModules;
    this.modulesToRead = new HashSet<>();
    this.promiseTypes = new HashMap<>();
  }

  public ReadModulesContext addModuleToRead(String moduleName) {
    if (!resolvedModules.containsKey(moduleName)) {
      modulesToRead.add(moduleName);
    }
    return this;
  }

  public boolean hasModulesToRead() {
    return !modulesToRead.isEmpty();
  }

  public String nextModuleToRead() {
    var moduleName = modulesToRead.iterator().next();
    modulesToRead.remove(moduleName);
    return moduleName;
  }

  public ReadModulesContext addResolvedModule(String moduleName, ModuleType moduleType) {
    if (moduleType != null) {
      resolvedModules.put(moduleName, moduleType);
    }
    modulesToRead.remove(moduleName);
    return this;
  }

  public PromiseType getOrCreatePromiseType(String fullyQualifiedName) {
    return promiseTypes.computeIfAbsent(fullyQualifiedName, PromiseType::new);
  }

  public Map<String, ModuleType> resolvedModules() {
    return resolvedModules;
  }

  public Map<String, PromiseType> promiseTypes() {
    return promiseTypes;
  }
}
