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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.python.cfg.fixpoint.ReadWriteVisitor.SymbolReadWrite;
import org.sonar.plugins.python.api.symbols.Symbol;

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
      for (Map<Symbol, SymbolReadWrite> symbolVariableUsageMap : liveVariables.variableReadWritesPerElement.values()) {
        for (Map.Entry<Symbol, SymbolReadWrite> symbolWithUsage : symbolVariableUsageMap.entrySet()) {
          if (symbolWithUsage.getValue().isRead()) {
            readAtLeastOnce.add(symbolWithUsage.getKey());
          }
        }
      }
    }
    return readAtLeastOnce;
  }

  public static class LiveVariables extends CfgBlockState {
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
      super(block);
    }

    /**
     * Builds a new LiveVariables instance for the given block and initializes the 'kill' and 'gen' symbol sets.
     */
    public static LiveVariables build(CfgBlock block) {
      LiveVariables instance = new LiveVariables(block);
      instance.init(block);
      return instance;
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


    public Set<Symbol> getIn() {
      return in;
    }

    public Set<Symbol> getOut() {
      return out;
    }
  }
}
