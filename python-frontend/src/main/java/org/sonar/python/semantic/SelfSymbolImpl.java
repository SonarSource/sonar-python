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
package org.sonar.python.semantic;

import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Name;

class SelfSymbolImpl extends SymbolImpl {

  private final Scope classScope;

  SelfSymbolImpl(String name, Scope classScope) {
    super(name, null);
    this.classScope = classScope;
  }

  @Override
  void addOrCreateChildUsage(Name nameTree, Usage.Kind kind) {
    SymbolImpl symbol = classScope.instanceAttributesByName.computeIfAbsent(nameTree.name(), name -> new SymbolImpl(name, null));
    symbol.addUsage(nameTree, kind);
  }

  @Override
  public void removeUsages() {
    super.removeUsages();
    classScope.instanceAttributesByName.values().forEach(SymbolImpl::removeUsages);
  }
}
