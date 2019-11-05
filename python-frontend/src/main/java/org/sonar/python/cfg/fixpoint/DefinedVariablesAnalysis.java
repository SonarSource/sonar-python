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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.python.semantic.Symbol;
import org.sonar.python.semantic.Usage;

public class DefinedVariablesAnalysis {

  private final Map<CfgBlock, DefinedVariables> definedVariablesPerBlock = new HashMap<>();

  public static DefinedVariablesAnalysis analyze(ControlFlowGraph cfg, Set<Symbol> localVariables) {
    DefinedVariablesAnalysis instance = new DefinedVariablesAnalysis();
    instance.compute(cfg, localVariables);
    return instance;
  }

  private void compute(ControlFlowGraph cfg, Set<Symbol> localVariables) {
    Map<Symbol, VariableDefinition> initialState = new HashMap<>();
    localVariables.forEach(variable -> {
      if (variable.usages().stream().anyMatch(u -> u.kind() == Usage.Kind.PARAMETER)) {
        initialState.put(variable, VariableDefinition.DEFINED);
        return;
      }
      initialState.put(variable, VariableDefinition.UNDEFINED);
    });
    Set<CfgBlock> blocks = cfg.blocks();
    blocks.forEach(block -> definedVariablesPerBlock.put(block, DefinedVariables.build(block, initialState)));
    Deque<CfgBlock> workList = new ArrayDeque<>(blocks);
    while (!workList.isEmpty()) {
      CfgBlock currentBlock = workList.pop();
      DefinedVariables definedVariables = this.definedVariablesPerBlock.get(currentBlock);
      boolean outHasChanged = definedVariables.propagate(this.definedVariablesPerBlock);
      if (outHasChanged) {
        currentBlock.successors().forEach(workList::push);
      }
    }
  }

  public DefinedVariables getDefinedVariables(CfgBlock block) {
    return definedVariablesPerBlock.get(block);
  }

  public enum VariableDefinition {
    BOTTOM,
    UNDEFINED,
    DEFINED,
    TOP;

    static VariableDefinition join(VariableDefinition v1, VariableDefinition v2) {
      if (v1 == BOTTOM) {
        return v2;
      }
      if (v2 == BOTTOM) {
        return v1;
      }
      if (v1 != v2) {
        return TOP;
      }
      return v1;
    }
  }

  public static class DefinedVariables extends CfgBlockState {

    private Map<Symbol, VariableDefinition> in = new HashMap<>();

    private Map<Symbol, VariableDefinition> out = new HashMap<>();

    private DefinedVariables(CfgBlock block) {
      super(block);
    }

    /**
     * Builds a new DefinedVariables instance for the given block and initializes the 'kill' symbol sets.
     */
    public static DefinedVariables build(CfgBlock block, Map<Symbol, VariableDefinition> initialState) {
      DefinedVariables instance = new DefinedVariables(block);
      instance.in = initialState;
      instance.init(block);
      return instance;
    }

    /**
     * Propagates forward: first computes the in set from all predecessors, then the out set.
     */
    private boolean propagate(Map<CfgBlock, DefinedVariables> definedVariablesPerBlock) {
      block.predecessors().stream()
        .map(definedVariablesPerBlock::get)
        .map(DefinedVariables::getOut)
        .forEach(predecessorOuts -> in = join(in, predecessorOuts));
      Map<Symbol, VariableDefinition> newOut = new HashMap<>(in);
      kill.forEach(symbol -> newOut.put(symbol, VariableDefinition.DEFINED));
      boolean outHasChanged = !newOut.equals(out);
      out = newOut;
      return outHasChanged;
    }

    private static Map<Symbol, VariableDefinition> join(Map<Symbol, VariableDefinition> programState1, Map<Symbol, VariableDefinition> programState2) {
      Map<Symbol, VariableDefinition> result = new HashMap<>();
      programState1.forEach((symbol, varDef) -> {
        VariableDefinition varDef2 = programState2.getOrDefault(symbol, VariableDefinition.BOTTOM);
        result.put(symbol, VariableDefinition.join(varDef, varDef2));
      });
      return result;
    }

    public Map<Symbol, VariableDefinition> getIn() {
      return in;
    }

    public Map<Symbol, VariableDefinition> getOut() {
      return out;
    }
  }
}
