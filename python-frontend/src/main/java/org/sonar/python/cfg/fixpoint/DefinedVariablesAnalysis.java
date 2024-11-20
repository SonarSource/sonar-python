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
package org.sonar.python.cfg.fixpoint;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;

public class DefinedVariablesAnalysis {

  private final Map<CfgBlock, DefinedVariables> definedVariablesPerBlock = new HashMap<>();

  public static DefinedVariablesAnalysis analyze(ControlFlowGraph cfg, Set<Symbol> localVariables) {
    DefinedVariablesAnalysis instance = new DefinedVariablesAnalysis();
    instance.compute(cfg, localVariables);
    return instance;
  }

  private void compute(ControlFlowGraph cfg, Set<Symbol> localVariables) {
    Map<Symbol, VariableDefinition> initialState = new HashMap<>();
    for (Symbol variable : localVariables) {
      boolean isParameter = variable.usages().stream().anyMatch(u -> u.kind() == Usage.Kind.PARAMETER);
      initialState.put(variable, isParameter ? VariableDefinition.DEFINED : VariableDefinition.UNDEFINED);
    }
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
    UNDEFINED,
    DEFINED;

    static VariableDefinition join(VariableDefinition v1, VariableDefinition v2) {
      if (v1 == UNDEFINED && v2 == UNDEFINED) {
        return UNDEFINED;
      }
      return DEFINED;
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
      Set<Symbol> allKeys = new HashSet<>(programState1.keySet());
      allKeys.addAll(programState2.keySet());
      for (Symbol key : allKeys) {
        VariableDefinition varDef1 = programState1.getOrDefault(key, VariableDefinition.UNDEFINED);
        VariableDefinition varDef2 = programState2.getOrDefault(key, VariableDefinition.UNDEFINED);
        result.put(key, VariableDefinition.join(varDef1, varDef2));
      }
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
