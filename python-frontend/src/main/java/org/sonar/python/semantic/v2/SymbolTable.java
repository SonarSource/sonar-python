/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.tree.Tree;

public record SymbolTable(Map<Tree, ScopeV2> scopesByRootTree) {

  public Set<SymbolV2> getSymbolsByRootTree(Tree tree) {
    return Optional.ofNullable(scopesByRootTree.get(tree))
      .map(ScopeV2::symbols)
      .map(Map::values)
      .map(HashSet::new)
      .orElseGet(HashSet::new);
  }

}
