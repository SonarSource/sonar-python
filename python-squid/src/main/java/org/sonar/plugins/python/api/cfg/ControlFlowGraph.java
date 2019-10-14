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
package org.sonar.plugins.python.api.cfg;

import java.util.Collections;
import java.util.Set;
import java.util.WeakHashMap;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.python.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.ControlFlowGraphBuilder;

public class ControlFlowGraph {

  private static final Logger LOG = Loggers.get(ControlFlowGraph.class);

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
}
