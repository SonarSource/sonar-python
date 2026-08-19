/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
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

  /**
   * Returns the blocks reachable from the control-flow graph start.
   * @param cfg the control-flow graph to traverse
   * @return the reachable blocks
   */
  public static Set<CfgBlock> reachableBlocks(ControlFlowGraph cfg) {
    Set<CfgBlock> reachableBlocks = new HashSet<>();
    Deque<CfgBlock> workList = new ArrayDeque<>();
    workList.push(cfg.start());
    while (!workList.isEmpty()) {
      CfgBlock currentBlock = workList.pop();
      if (reachableBlocks.add(currentBlock)) {
        currentBlock.successors().forEach(workList::push);
      }
    }
    return reachableBlocks;
  }

  /**
   * Returns the blocks that are not reachable from the control-flow graph start.
   * @param cfg the control-flow graph to inspect
   * @return the unreachable blocks
   */
  public static Set<CfgBlock> unreachableBlocks(ControlFlowGraph cfg) {
    return difference(cfg.blocks(), reachableBlocks(cfg));
  }

  private static Set<CfgBlock> difference(Set<CfgBlock> a, Set<CfgBlock> b) {
    Set<CfgBlock> result = new HashSet<>(a);
    result.removeAll(b);
    return result;
  }
}
