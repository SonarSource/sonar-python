/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.List;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.Tree;

public interface CfgBlock {

  Set<CfgBlock> successors();

  Set<CfgBlock> predecessors();

  List<Tree> elements();

  /**
   * @return block following this one if no jump is applied
   * Returns {@code null} if this block doesn't end with jump statement (break, continue, return, raise)
   */
  @CheckForNull
  CfgBlock syntacticSuccessor();
}
