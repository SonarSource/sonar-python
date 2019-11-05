/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.cfg.fixpoint;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.UsageVisitor.SymbolUsage;
import org.sonar.python.semantic.Symbol;

public abstract class CfgBlockState {

  protected final CfgBlock block;
  protected final Map<Tree, Map<Symbol, SymbolUsage>> variableUsagesPerElement;

  /**
   * variables that are being read in the block
   */
  protected final Set<Symbol> gen = new HashSet<>();

  /**
   * variables that are being written in the block
   */
  protected final Set<Symbol> kill = new HashSet<>();


  CfgBlockState(CfgBlock block) {
    this.block = block;
    this.variableUsagesPerElement = new HashMap<>();
  }

  public Map<Symbol, SymbolUsage> getVariableUsages(Tree tree) {
    return variableUsagesPerElement.get(tree);
  }

  protected void init(CfgBlock block) {
    // 'writtenOnly' has variables that are WRITE-ONLY inside at least one element
    // (as opposed to 'kill' which can have a variable that inside an element is both READ and WRITTEN)
    Set<Symbol> writtenOnly = new HashSet<>();
    for (Tree element : block.elements()) {
      UsageVisitor usageVisitor = new UsageVisitor();
      element.accept(usageVisitor);
      variableUsagesPerElement.put(element, usageVisitor.symbolToUsages());
      computeGenAndKill(writtenOnly, usageVisitor.symbolToUsages());
    }
  }

  /**
   * This has side effects on 'writtenOnly'
   */
  private void computeGenAndKill(Set<Symbol> writtenOnly, Map<Symbol, SymbolUsage> symbolToUsages) {
    for (Map.Entry<Symbol, SymbolUsage> symbolListEntry : symbolToUsages.entrySet()) {
      Symbol symbol = symbolListEntry.getKey();
      SymbolUsage usage = symbolListEntry.getValue();
      if (usage.isRead() && !writtenOnly.contains(symbol)) {
        gen.add(symbol);
      }
      if (usage.isWrite()) {
        kill.add(symbol);
        if (!usage.isRead()) {
          writtenOnly.add(symbol);
        }
      }
    }
  }

  public Set<Symbol> getGen() {
    return gen;
  }

  public Set<Symbol> getKill() {
    return kill;
  }
}
