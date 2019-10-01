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

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.python.api.tree.Tree;

public class PythonCfgBlock implements CfgBlock {

  private final LinkedList<Tree> elements = new LinkedList<>();
  private CfgBlock syntacticSuccessor;
  private Set<CfgBlock> successors;

  public PythonCfgBlock(CfgBlock successor) {
    this.successors = Collections.singleton(successor);
  }

  public PythonCfgBlock(Set<CfgBlock> successors) {
    this.successors = successors;
  }

  @Override
  public Set<CfgBlock> successors() {
    return successors;
  }

  @Override
  public Set<CfgBlock> predecessors() {
    return Collections.emptySet();
  }

  @Override
  public List<Tree> elements() {
    return elements;
  }

  public void addElement(Tree tree) {
    elements.addFirst(tree);
  }

  @CheckForNull
  @Override
  public CfgBlock syntacticSuccessor() {
    return syntacticSuccessor;
  }

  public void setSyntacticSuccessor(CfgBlock syntacticSuccessor) {
    this.syntacticSuccessor = syntacticSuccessor;
  }

  public boolean isEmptyBlock() {
    return elements.isEmpty() && successors.size() == 1;
  }

  PythonCfgBlock firstNonEmptySuccessor() {
    PythonCfgBlock block = this;
    while (block.isEmptyBlock()) {
      // TODO: handle loops with empty blocks
      block = (PythonCfgBlock) block.successors().iterator().next();
    }
    return block;
  }

  /**
   * Replace successors based on a replacement map.
   * This method is used when we remove empty blocks:
   * we have to replace empty successors in the remaining blocks by non-empty successors.
   */
  void replaceSuccessors(Map<PythonCfgBlock, PythonCfgBlock> replacements) {
    successors = successors.stream()
      .map(successor -> replacements.getOrDefault(successor, (PythonCfgBlock) successor))
      .collect(Collectors.toSet());
    if (syntacticSuccessor != null) {
      syntacticSuccessor = replacements.getOrDefault(syntacticSuccessor, (PythonCfgBlock) syntacticSuccessor);
    }
  }

}
