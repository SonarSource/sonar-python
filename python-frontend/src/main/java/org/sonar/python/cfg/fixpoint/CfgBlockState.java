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
package org.sonar.python.cfg.fixpoint;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReadWriteVisitor.SymbolReadWrite;
import org.sonar.plugins.python.api.symbols.Symbol;

public abstract class CfgBlockState {

  protected final CfgBlock block;
  protected final Map<Tree, Map<Symbol, SymbolReadWrite>> variableReadWritesPerElement;

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
    this.variableReadWritesPerElement = new HashMap<>();
  }

  public Map<Symbol, SymbolReadWrite> getSymbolReadWrites(Tree tree) {
    return variableReadWritesPerElement.get(tree);
  }

  protected void init(CfgBlock block) {
    // 'writtenOnly' has variables that are WRITE-ONLY inside at least one element
    // (as opposed to 'kill' which can have a variable that inside an element is both READ and WRITTEN)
    Set<Symbol> writtenOnly = new HashSet<>();
    for (Tree element : block.elements()) {
      ReadWriteVisitor readWriteVisitor = new ReadWriteVisitor();
      element.accept(readWriteVisitor);
      variableReadWritesPerElement.put(element, readWriteVisitor.symbolToUsages());
      computeGenAndKill(writtenOnly, readWriteVisitor.symbolToUsages());
    }
  }

  /**
   * This has side effects on 'writtenOnly'
   */
  private void computeGenAndKill(Set<Symbol> writtenOnly, Map<Symbol, SymbolReadWrite> symbolToUsages) {
    for (Map.Entry<Symbol, SymbolReadWrite> symbolListEntry : symbolToUsages.entrySet()) {
      Symbol symbol = symbolListEntry.getKey();
      SymbolReadWrite usage = symbolListEntry.getValue();
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

  public boolean isSymbolUsedInBlock(Symbol symbol) {
    return variableReadWritesPerElement.values().stream()
      .flatMap(m -> m.keySet().stream())
      .anyMatch(s -> s == symbol);
  }
}
