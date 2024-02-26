/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;

public class CfgUtils {

  private CfgUtils() {
    // empty constructor
  }

  public static Set<CfgBlock> unreachableBlocks(ControlFlowGraph cfg) {
    Set<CfgBlock> reachableBlocks = new HashSet<>();
    Deque<CfgBlock> workList = new ArrayDeque<>();
    workList.push(cfg.start());
    while (!workList.isEmpty()) {
      CfgBlock currentBlock = workList.pop();
      if (reachableBlocks.add(currentBlock)) {
        currentBlock.successors().forEach(workList::push);
      }
    }
    return difference(cfg.blocks(), reachableBlocks);
  }

  private static Set<CfgBlock> difference(Set<CfgBlock> a, Set<CfgBlock> b) {
    Set<CfgBlock> result = new HashSet<>(a);
    result.removeAll(b);
    return result;
  }
}
