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

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.api.internal.google.common.collect.Lists;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.python.api.tree.ReturnStatement;
import org.sonar.python.api.tree.Statement;
import org.sonar.python.api.tree.StatementList;

public class ControlFlowGraphBuilder {

  private final CfgBlock start;
  private final CfgBlock end = new PythonCfgEndBlock();
  private final Set<CfgBlock> blocks = new HashSet<>();

  public ControlFlowGraphBuilder(@Nullable StatementList statementList) {
    blocks.add(end);
    if (statementList != null) {
      start = build(statementList.statements(), createSimpleBlock(end));
    } else {
      start = end;
    }
  }

  public ControlFlowGraph getCfg() {
    return new ControlFlowGraph(blocks, start, end);
  }

  private PythonCfgBlock createSimpleBlock(CfgBlock successor) {
    PythonCfgBlock block = new PythonCfgBlock(successor);
    blocks.add(block);
    return block;
  }

  private PythonCfgBlock build(List<Statement> statements, PythonCfgBlock successor) {
    PythonCfgBlock currentBlock = successor;
    for (Statement statement : Lists.reverse(statements)) {
      currentBlock = build(statement, currentBlock);
    }
    return currentBlock;
  }

  private PythonCfgBlock build(Statement statement, PythonCfgBlock currentBlock) {
    switch (statement.getKind()) {
      case RETURN_STMT:
        return buildReturnStatement((ReturnStatement) statement, currentBlock);
      default:
        currentBlock.addElement(statement);
    }

    return currentBlock;
  }

  private PythonCfgBlock buildReturnStatement(ReturnStatement statement, PythonCfgBlock syntacticSuccessor) {
    if (syntacticSuccessor.isEmptyBlock()) {
      syntacticSuccessor.addElement(statement);
      syntacticSuccessor.setSyntacticSuccessor(syntacticSuccessor.successors().iterator().next());
      return syntacticSuccessor;
    } else {
      PythonCfgBlock block = createSimpleBlock(end);
      block.setSyntacticSuccessor(syntacticSuccessor);
      block.addElement(statement);
      return block;
    }
  }

}
