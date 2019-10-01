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

import java.util.Arrays;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.cfg.PythonCfgBlock;

import static org.assertj.core.api.Assertions.assertThat;

public class ControlFlowGraphTest {

  @Test
  public void empty_file() {
    ControlFlowGraph cfg = fileCfg("");
    CfgBlock start = cfg.start();
    assertThat(start).isEqualTo(cfg.end());
    assertThat(start.elements()).isEmpty();
    assertThat(start.successors()).isEmpty();
    assertThat(start.predecessors()).isEmpty();
    assertThat(start.syntacticSuccessor()).isNull();
    assertThat(cfg.blocks()).containsExactly(start);
  }

  @Test
  public void pass_statement() {
    ControlFlowGraph cfg = cfg("pass");
    CfgBlock start = cfg.start();
    assertThat(start.elements()).extracting(Tree::getKind).containsExactly(Kind.PASS_STMT);
    assertThat(start.successors()).containsExactly(cfg.end());
    assertThat(start.predecessors()).isEmpty();
    assertThat(start.syntacticSuccessor()).isNull();
    assertThat(cfg.blocks()).containsExactlyInAnyOrder(start, cfg.end());
  }

  @Test
  public void single_element() {
    ControlFlowGraph cfg = verifyCfg("b1(succ = [END], pred = [])");
    assertThat(cfg.blocks()).containsExactlyInAnyOrder(cfg.start(), cfg.end());
  }

  @Test
  public void element_order() {
    ControlFlowGraph cfg = verifyCfg("b1(succ = [END], pred = []); pass");
    assertThat(cfg.start().elements()).extracting(Tree::getKind).containsExactly(Kind.EXPRESSION_STMT, Kind.PASS_STMT);
  }

  @Test
  public void return_statement() {
    verifyCfg(
      "b1(succ = [END], syntSucc = END, pred = [])",
      "return");
    ControlFlowGraph cfg = verifyCfg(
      "b1(succ = [END], syntSucc = b2, pred = [])",
      "return",
      "b2(succ = [END], pred = [])");
    assertThat(cfg.start().elements().get(0).firstToken().value()).isEqualTo("b1");
  }

  private ControlFlowGraph verifyCfg(String... lines) {
    ControlFlowGraph cfg = cfg(lines);
    CfgValidator.assertCfgStructure(cfg);
    assertNoEmptyBlocksInCFG(cfg);
    return cfg;
  }

  private void assertNoEmptyBlocksInCFG(ControlFlowGraph cfg) {
    cfg.blocks().stream()
      .filter(block -> block instanceof PythonCfgBlock)
      .forEach(block -> assertThat(((PythonCfgBlock) block).isEmptyBlock()).isFalse());
  }

  private ControlFlowGraph cfg(String... lines) {
    FileInput fileInput = PythonTestUtils.parse("def f():", Arrays.stream(lines).map(s -> "  " + s).collect(Collectors.joining("\n")));
    FunctionDef fun = (FunctionDef) fileInput.descendants(Kind.FUNCDEF).findFirst().get();
    return ControlFlowGraph.build(fun);
  }

  private ControlFlowGraph fileCfg(String... lines) {
    FileInput fileInput = PythonTestUtils.parse(lines);
    return ControlFlowGraph.build(fileInput);
  }

}
