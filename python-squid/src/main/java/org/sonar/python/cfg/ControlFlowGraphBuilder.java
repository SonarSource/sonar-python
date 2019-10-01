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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.python.api.tree.IfStatement;
import org.sonar.python.api.tree.ReturnStatement;
import org.sonar.python.api.tree.Statement;
import org.sonar.python.api.tree.StatementList;

public class ControlFlowGraphBuilder {

  private PythonCfgBlock start;
  private final PythonCfgBlock end = new PythonCfgEndBlock();
  private final Set<PythonCfgBlock> blocks = new HashSet<>();

  public ControlFlowGraphBuilder(@Nullable StatementList statementList) {
    blocks.add(end);
    if (statementList != null) {
      start = build(statementList.statements(), createSimpleBlock(end));
    } else {
      start = end;
    }
    removeEmptyBlocks();
  }

  private void removeEmptyBlocks() {
    Map<PythonCfgBlock, PythonCfgBlock> emptyBlockReplacements = new HashMap<>();
    for (PythonCfgBlock block : blocks) {
      if (block.isEmptyBlock()) {
        PythonCfgBlock firstNonEmptySuccessor = block.firstNonEmptySuccessor();
        emptyBlockReplacements.put(block, firstNonEmptySuccessor);
      }
    }

    blocks.removeAll(emptyBlockReplacements.keySet());

    for (PythonCfgBlock block : blocks) {
      block.replaceSuccessors(emptyBlockReplacements);
    }

    start = emptyBlockReplacements.getOrDefault(start, start);
  }

  public ControlFlowGraph getCfg() {
    return new ControlFlowGraph(Collections.unmodifiableSet(blocks), start, end);
  }

  private PythonCfgBlock createSimpleBlock(CfgBlock successor) {
    PythonCfgBlock block = new PythonCfgBlock(successor);
    blocks.add(block);
    return block;
  }

  private PythonCfgBlock createSimpleBlock(CfgBlock... successors) {
    PythonCfgBlock block = new PythonCfgBlock(new HashSet<>(Arrays.asList(successors)));
    blocks.add(block);
    return block;
  }

  private PythonCfgBlock build(List<Statement> statements, PythonCfgBlock successor) {
    PythonCfgBlock currentBlock = successor;
    for (int i = statements.size() - 1; i >= 0; i--) {
      Statement statement = statements.get(i);
      currentBlock = build(statement, currentBlock);
    }
    return currentBlock;
  }

  private PythonCfgBlock build(Statement statement, PythonCfgBlock currentBlock) {
    switch (statement.getKind()) {
      case RETURN_STMT:
        return buildReturnStatement((ReturnStatement) statement, currentBlock);
      case IF_STMT:
        return buildIfStatement(((IfStatement) statement), currentBlock);
      default:
        currentBlock.addElement(statement);
    }

    return currentBlock;
  }

  private PythonCfgBlock buildIfStatement(IfStatement ifStatement, PythonCfgBlock currentBlock) {
    PythonCfgBlock ifBodyBlock = createSimpleBlock(currentBlock);
    ifBodyBlock = build(ifStatement.body().statements(), ifBodyBlock);
    PythonCfgBlock beforeIfBlock = createSimpleBlock(ifBodyBlock, currentBlock);
    beforeIfBlock.addElement(ifStatement.condition());
    return beforeIfBlock;
  }

  private PythonCfgBlock buildReturnStatement(ReturnStatement statement, PythonCfgBlock syntacticSuccessor) {
    PythonCfgBlock block = createSimpleBlock(end);
    block.setSyntacticSuccessor(syntacticSuccessor);
    block.addElement(statement);
    return block;
  }

}
