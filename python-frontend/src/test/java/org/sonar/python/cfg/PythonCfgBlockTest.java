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

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class PythonCfgBlockTest {

  @Test
  void loop_of_empty_block() {
    PythonCfgEndBlock end = new PythonCfgEndBlock();
    PythonCfgSimpleBlock succ = new PythonCfgSimpleBlock(end);
    PythonCfgSimpleBlock block = new PythonCfgSimpleBlock(succ);
    PythonCfgSimpleBlock start = new PythonCfgSimpleBlock(block);
    Map<PythonCfgBlock, PythonCfgBlock> replacements = new HashMap<>();
    replacements.put(end, block);
    succ.replaceSuccessors(replacements);
    assertThat(start.firstNonEmptySuccessor()).isEqualTo(succ);
  }

  @Test
  void displaysUnknownPositionWhenFirstTokenIsMissing() {
    Tree tree = mock(Tree.class);
    PythonCfgSimpleBlock block = new PythonCfgSimpleBlock(new PythonCfgEndBlock());
    block.addElement(tree);
    PythonCfgBranchingBlock branchingBlock = new PythonCfgBranchingBlock(tree, null, new PythonCfgEndBlock());

    assertThat(block.toStringDisplayPosition()).isEqualTo("unknown:");
    assertThat(branchingBlock.toStringDisplayPosition()).isEqualTo("unknown:");
  }
}
