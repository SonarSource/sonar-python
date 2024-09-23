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

import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.converter.PythonTypeToDescriptorConverter;

public record SymbolTable(Map<Tree, ScopeV2> scopesByRootTree) {

  public Set<SymbolV2> getSymbolsByRootTree(Tree tree) {
    return Optional.ofNullable(scopesByRootTree.get(tree))
      .map(ScopeV2::symbols)
      .map(Map::values)
      .map(HashSet::new)
      .orElseGet(HashSet::new);
  }

  Set<Descriptor> convertScopeSymbolsToDescriptor(Tree tree) {

    var topLevelSymbols = getSymbolsByRootTree(tree);
    var result = new HashSet<Descriptor>();

    for (var symbol : topLevelSymbols) {
//      symbol.usages().get(0).

      var usages = symbol.usages()
        .stream()
        .filter(UsageV2::isBindingUsage)
        .map(UsageV2::tree)
        .map(tree1 -> {
          if (tree1 instanceof Expression expression) {
            return expression.typeV2();
          }
          return null;
        })
        .filter(Objects::nonNull)
        .toList();

      // Here, we assume that there is only 1 type
      var type = usages.get(0);

      var converted = PythonTypeToDescriptorConverter.convert(type, symbol);
      if (converted != null) {
        result.add(converted);
      }
    }
    return result;
  }
}
