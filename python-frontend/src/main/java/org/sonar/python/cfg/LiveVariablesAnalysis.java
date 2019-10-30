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
package org.sonar.python.cfg;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.Symbol;
import org.sonar.python.semantic.Usage;

public class LiveVariablesAnalysis {

  private final Map<CfgBlock, LiveVariables> liveVariablesPerBlock = new HashMap<>();

  public static LiveVariablesAnalysis analyze(ControlFlowGraph cfg) {
    LiveVariablesAnalysis instance = new LiveVariablesAnalysis();
    instance.compute(cfg);
    return instance;
  }

  /**
   * See "worklist algorithm" in http://www.cs.cornell.edu/courses/cs4120/2013fa/lectures/lec26-fa13.pdf
   * An alternative terminology for "kill/gen" is "def/use"
   */
  private void compute(ControlFlowGraph cfg) {
    cfg.blocks().forEach(block -> liveVariablesPerBlock.put(block, LiveVariables.build(block)));
    Deque<CfgBlock> workList = new ArrayDeque<>(cfg.blocks());
    while (!workList.isEmpty()) {
      CfgBlock currentBlock = workList.pop();
      LiveVariables liveVariables = liveVariablesPerBlock.get(currentBlock);
      boolean liveInHasChanged = liveVariables.propagate(liveVariablesPerBlock);
      if (liveInHasChanged) {
        currentBlock.predecessors().forEach(workList::push);
      }
    }
  }

  public LiveVariables getLiveVariables(CfgBlock block) {
    return liveVariablesPerBlock.get(block);
  }

  public Set<Symbol> getReadSymbols() {
    Set<Symbol> readAtLeastOnce = new HashSet<>();
    for (LiveVariables liveVariables : liveVariablesPerBlock.values()) {
      for (Map<Symbol, SymbolUsage> symbolVariableUsageMap : liveVariables.variableUsagesPerElement.values()) {
        for (Map.Entry<Symbol, SymbolUsage> symbolWithUsage : symbolVariableUsageMap.entrySet()) {
          if (symbolWithUsage.getValue().isRead()) {
            readAtLeastOnce.add(symbolWithUsage.getKey());
          }
        }
      }
    }
    return readAtLeastOnce;
  }

  public static final class SymbolUsage {
    private boolean isRead = false;
    private boolean isWrite = false;

    public boolean isWrite() {
      return isWrite;
    }

    public boolean isRead() {
      return isRead;
    }
  }

  public static class LiveVariables {

    private final CfgBlock block;
    private final Map<Tree, Map<Symbol, SymbolUsage>> variableUsagesPerElement;

    /**
     * variables that are being read in the block
     */
    private final Set<Symbol> gen = new HashSet<>();

    /**
     * variables that are being written in the block
     */
    private final Set<Symbol> kill = new HashSet<>();

    /**
     * The Live-In variables are variables which has values that:
     * - are needed by this block
     * OR
     * - are needed by a successor block and are not killed in this block.
     */
    private Set<Symbol> in = new HashSet<>();

    /**
     * The Live-Out variables are variables which are needed by successors.
     */
    private Set<Symbol> out = new HashSet<>();

    private LiveVariables(CfgBlock block) {
      this.block = block;
      this.variableUsagesPerElement = new HashMap<>();
    }

    public Map<Symbol, SymbolUsage> getVariableUsages(Tree tree) {
      return variableUsagesPerElement.get(tree);
    }

    /**
     * Builds a new LiveVariables instance for the given block and initializes the 'kill' and 'gen' symbol sets.
     */
    public static LiveVariables build(CfgBlock block) {
      LiveVariables instance = new LiveVariables(block);
      instance.init(block);
      return instance;
    }

    private void init(CfgBlock block) {
      // 'writtenOnly' has variables that are WRITE-ONLY inside at least one element
      // (as opposed to 'kill' which can have a variable that inside an element is both READ and WRITTEN)
      Set<Symbol> writtenOnly = new HashSet<>();
      for (Tree element : block.elements()) {
        UsageVisitor usageVisitor = new UsageVisitor();
        usageVisitor.scan(element);
        variableUsagesPerElement.put(element, usageVisitor.symbolToUsages);
        computeGenAndKill(writtenOnly, usageVisitor.symbolToUsages);
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

    /**
     * Propagates backwards: first computes the 'out' set, then the 'in' set.
     */
    private boolean propagate(Map<CfgBlock, LiveVariables> liveVariablesPerBlock) {
      out.clear();
      block.successors().stream()
        .map(liveVariablesPerBlock::get)
        .map(LiveVariables::getIn)
        .forEach(out::addAll);
      // in = gen + (out - kill)
      Set<Symbol> newIn = new HashSet<>(gen);
      newIn.addAll(difference(out, kill));
      boolean inHasChanged = !newIn.equals(in);
      in = newIn;
      return inHasChanged;
    }

    private static Set<Symbol> difference(Set<Symbol> out, Set<Symbol> kill) {
      Set<Symbol> result = new HashSet<>(out);
      result.removeIf(kill::contains);
      return result;
    }

    public Set<Symbol> getGen() {
      return gen;
    }

    public Set<Symbol> getKill() {
      return kill;
    }

    public Set<Symbol> getIn() {
      return in;
    }

    public Set<Symbol> getOut() {
      return out;
    }
  }

  private static class UsageVisitor extends BaseTreeVisitor {
    private Map<Symbol, SymbolUsage> symbolToUsages = new HashMap<>();

    @Override
    protected void scan(@Nullable Tree tree) {
      getSymbol(tree).ifPresent(symbol ->
        symbol.usages().stream()
          .filter(usage -> usage.tree() == tree)
          .findFirst()
          .ifPresent(usage -> addToSymbolToUsageMap(usage, symbol))
      );
      super.scan(tree);
    }

    private void addToSymbolToUsageMap(Usage usage, Symbol symbol) {
      SymbolUsage symbolUsage = symbolToUsages.getOrDefault(symbol, new SymbolUsage());
      if (!usage.isBindingUsage()) {
        symbolUsage.isRead = true;
      } else if (usage.kind() == Usage.Kind.COMPOUND_ASSIGNMENT_LHS) {
        symbolUsage.isRead = true;
        symbolUsage.isWrite = true;
      } else {
        symbolUsage.isWrite = true;
      }
      symbolToUsages.put(symbol, symbolUsage);
    }

    private static Optional<Symbol> getSymbol(@Nullable Tree tree) {
      if (tree instanceof HasSymbol) {
        return Optional.ofNullable(((HasSymbol) tree).symbol());
      }
      return Optional.empty();
    }
  }
}
