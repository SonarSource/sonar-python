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
package org.sonar.plugins.python.api.cfg;

import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.function.Function;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.ControlFlowGraphBuilder;

public class ControlFlowGraph {

  private static final Logger LOG = LoggerFactory.getLogger(ControlFlowGraph.class);

  private final Set<CfgBlock> blocks;
  private final CfgBlock start;
  private final CfgBlock end;

  // we shouldn't prevent trees from being garbage collected
  private static Set<Tree> treesWithCfgErrors = Collections.newSetFromMap(new WeakHashMap<>());

  public ControlFlowGraph(Set<CfgBlock> blocks, CfgBlock start, CfgBlock end) {
    this.blocks = blocks;
    this.start = start;
    this.end = end;
  }

  @CheckForNull
  private static ControlFlowGraph build(@Nullable StatementList statementList, PythonFile file) {
    if (!treesWithCfgErrors.contains(statementList)) {
      try {
        return new ControlFlowGraphBuilder(statementList).getCfg();
      } catch (Exception e) {
        treesWithCfgErrors.add(statementList);
        LOG.warn("Failed to build control flow graph in file [{}]: {}", file, e.getMessage());
      }
    }
    return null;
  }

  @CheckForNull
  public static ControlFlowGraph build(FunctionDef functionDef, PythonFile file) {
    return build(functionDef.body(), file);
  }

  @CheckForNull
  public static ControlFlowGraph build(FileInput fileInput, PythonFile file) {
    return build(fileInput.statements(), file);
  }

  public CfgBlock start() {
    return start;
  }

  public CfgBlock end() {
    return end;
  }

  public Set<CfgBlock> blocks() {
    return blocks;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();

    Function<Collection<CfgBlock>, List<CfgBlock>> sortedByDisplayString = list -> list.stream()
      .sorted(Comparator.comparing(Object::toString))
      .toList();

    List<CfgBlock> sortedBlocks = sortedByDisplayString.apply(blocks);
    int graphNodeId = 0;
    Map<CfgBlock, Integer> graphNodeIds = new HashMap<>();
    for (CfgBlock block : sortedBlocks) {
      graphNodeIds.put(block, graphNodeId);
      sb.append(graphNodeId).append("[label=\"").append(block.toString()).append("\"];");
      graphNodeId++;
    }
    for (CfgBlock block : sortedBlocks) {
      int id = graphNodeIds.get(block);
      for (CfgBlock successor : sortedByDisplayString.apply(block.successors())) {
        sb.append(id).append("->").append(graphNodeIds.get(successor)).append(";");
      }
      if (block.syntacticSuccessor() != null) {
        sb.append(id).append("->").append(graphNodeIds.get(block.syntacticSuccessor())).append("[style=dotted];");
      }
    }
    return sb.toString();
  }

}
