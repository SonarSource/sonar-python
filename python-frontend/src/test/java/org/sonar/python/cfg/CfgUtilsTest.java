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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class CfgUtilsTest {

  private final PythonFile file = mock(PythonFile.class, "file1.py");

  @Test
  void all_blocks_reachable_without_jumps() {
    ControlFlowGraph cfg = cfg(
      "x = 10"
    );

    assertAllBlocksReachable(cfg);
  }

  @Test
  void return_makes_its_syntactic_successor_unreachable() {
    ControlFlowGraph cfg = cfg(
      "x = 10",
      "return",
      "y = 42"
    );

    assertOnlySyntacticSuccessorIsUnreachable(cfg);
  }

  @Test
  void raise_makes_its_syntactic_successor_unreachable() {
    ControlFlowGraph cfg = cfg(
      "x = 10",
      "raise RuntimeError()",
      "y = 42"
    );

    assertOnlySyntacticSuccessorIsUnreachable(cfg);
  }

  @Test
  void break_makes_its_syntactic_successor_unreachable() {
    ControlFlowGraph cfg = cfg(
      "while condition:",
      "  break",
      "  unreachable()",
      "after_loop()"
    );

    assertOnlySyntacticSuccessorIsUnreachable(cfg);
  }

  @Test
  void continue_makes_its_syntactic_successor_unreachable() {
    ControlFlowGraph cfg = cfg(
      "while condition:",
      "  continue",
      "  unreachable()",
      "after_loop()"
    );

    assertOnlySyntacticSuccessorIsUnreachable(cfg);
  }

  @Test
  void multiple_blocks_reachable_with_branching() {
    ControlFlowGraph cfg = cfg(
      "if condition:",
      "  if_branch()",
      "else:",
      "  else_branch()",
      "after_branch()"
    );

    assertThat(cfg.start().successors()).hasSize(2);
    assertAllBlocksReachable(cfg);
  }

  private static void assertAllBlocksReachable(ControlFlowGraph cfg) {
    assertThat(CfgUtils.reachableBlocks(cfg)).isEqualTo(cfg.blocks());
    assertThat(CfgUtils.unreachableBlocks(cfg)).isEmpty();
  }

  private static void assertOnlySyntacticSuccessorIsUnreachable(ControlFlowGraph cfg) {
    Set<CfgBlock> syntacticSuccessors = cfg.blocks().stream()
      .map(CfgBlock::syntacticSuccessor)
      .filter(Objects::nonNull)
      .collect(Collectors.toSet());
    assertThat(syntacticSuccessors).hasSize(1);

    Set<CfgBlock> expectedReachableBlocks = new HashSet<>(cfg.blocks());
    expectedReachableBlocks.removeAll(syntacticSuccessors);
    assertThat(CfgUtils.reachableBlocks(cfg)).isEqualTo(expectedReachableBlocks);
    assertThat(CfgUtils.unreachableBlocks(cfg)).isEqualTo(syntacticSuccessors);
  }

  private ControlFlowGraph cfg(String... lines) {
    FileInput fileInput = PythonTestUtils.parse("def wrapper():", Arrays.stream(lines).map(s -> "  " + s).collect(Collectors.joining("\n")));
    FunctionDef fun = (FunctionDef) fileInput.statements().statements().get(0);
    return ControlFlowGraph.build(fun, file);
  }
}
