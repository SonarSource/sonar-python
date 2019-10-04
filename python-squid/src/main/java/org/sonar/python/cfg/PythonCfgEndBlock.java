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
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.python.api.tree.Tree;

public class PythonCfgEndBlock extends PythonCfgBlock {

  @Override
  public Set<CfgBlock> successors() {
    return Collections.emptySet();
  }

  @Override
  public Set<CfgBlock> predecessors() {
    return Collections.emptySet();
  }

  @Override
  public List<Tree> elements() {
    return Collections.emptyList();
  }

  @Override
  void replaceSuccessors(Map<PythonCfgBlock, PythonCfgBlock> replacements) {
    // nothing to do
  }

  @CheckForNull
  @Override
  public CfgBlock syntacticSuccessor() {
    return null;
  }

  @Override
  public String toString() {
    return "END";
  }
}
