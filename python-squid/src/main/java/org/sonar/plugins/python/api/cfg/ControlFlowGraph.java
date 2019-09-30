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

import java.util.Set;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.cfg.ControlFlowGraphBuilder;

public class ControlFlowGraph {

  private final Set<CfgBlock> blocks;
  private final CfgBlock start;
  private final CfgBlock end;

  public ControlFlowGraph(Set<CfgBlock> blocks, CfgBlock start, CfgBlock end) {
    this.blocks = blocks;
    this.start = start;
    this.end = end;
  }

  public static ControlFlowGraph build(FunctionDef functionDef) {
    return new ControlFlowGraphBuilder(functionDef.body()).getCfg();
  }

  public static ControlFlowGraph build(FileInput fileInput) {
    return new ControlFlowGraphBuilder(fileInput.statements()).getCfg();
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
