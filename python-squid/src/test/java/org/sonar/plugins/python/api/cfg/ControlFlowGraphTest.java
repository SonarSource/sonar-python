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
import org.sonar.python.cfg.PythonCfgEndBlock;

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

  @Test
  public void if_statement() {
    verifyCfg(
      "before(succ = [if_body, END], elem = 2)",
      "if cond:",
      "  if_body(succ = [END], elem = 1)"
    );

    verifyCfg(
      "before(succ = [if_body, after], elem = 2)",
      "if cond:",
      "  if_body(succ = [after], elem = 1)",
      "after(succ = [END], elem = 1)"
    );
  }

  @Test
  public void if_statement_with_return() {
    verifyCfg(
      "before(succ = [if_body, END])",
      "if cond:",
      "  if_body(succ = [END], syntSucc = END)",
      "  return"
    );

    verifyCfg(
      "before(succ = [if_body, after])",
      "if cond:",
      "  if_body(succ = [END], syntSucc = after)",
      "  return",
      "after(succ = [END])"
    );
  }

  @Test
  public void if_else_statement() {
    verifyCfg(
      "before(succ = [if_body, else_body], elem = 2)",
      "if cond:",
      "  if_body(succ = [END], elem = 1)",
      "else:",
      "  else_body(succ = [END], elem = 1)"
    );

    verifyCfg(
      "before(succ = [if_body, else_body], elem = 2)",
      "if cond:",
      "  if_body(succ = [after], elem = 1)",
      "else:",
      "  else_body(succ = [after], elem = 1)",
      "after(succ = [END], elem = 1)"
    );
  }

  @Test
  public void if_else_statement_with_return() {
    verifyCfg(
      "before(succ = [if_body, else_body], elem = 2)",
      "if cond:",
      "  if_body(succ = [END], elem = 2, syntSucc = END)",
      "  return",
      "else:",
      "  else_body(succ = [END], elem = 1)"
    );

    verifyCfg(
      "before(succ = [if_body, else_body])",
      "if cond:",
      "  if_body(succ = [END], syntSucc = END)",
      "  return",
      "else:",
      "  else_body(succ = [END], syntSucc = END)",
      "  return"
    );
  }

  @Test
  public void if_elif_statement() {
    verifyCfg(
      "before(succ = [if_body, before_elif_body], elem = 2)",
      "if cond1:",
      "  if_body(succ = [END], elem = 1)",
      "elif before_elif_body(succ = [elif_body, END], elem = 1):",
      "  elif_body(succ = [END], elem = 1)"
    );

    verifyCfg(
      "before(succ = [if_body, before_elif_body], elem = 2)",
      "if cond1:",
      "  if_body(succ = [after], elem = 1)",
      "elif before_elif_body(succ = [elif_body, after], elem = 1):",
      "  elif_body(succ = [after], elem = 1)",
      "after(succ = [END], elem = 1)"
    );

    verifyCfg(
      "before(succ = [if_body, before_elif_body_1], elem = 2)",
      "if cond1:",
      "  if_body(succ = [END], elem = 1)",
      "elif before_elif_body_1(succ = [elif_body_1, before_elif_body_2], elem = 1):",
      "  elif_body_1(succ = [END], elem = 1)",
      "elif before_elif_body_2(succ = [elif_body_2, END], elem = 1):",
      "  elif_body_2(succ = [END], elem = 1)"
    );
  }

  @Test
  public void if_elif_else_statement() {
    verifyCfg(
      "before(succ = [if_body, before_elif_body_1], elem = 2)",
      "if cond1:",
      "  if_body(succ = [END], elem = 1)",
      "elif before_elif_body_1(succ = [else_body, elif_body_1], elem = 1):",
      "  elif_body_1(succ = [END], elem = 1)",
      "else:",
      "  else_body(succ = [END], elem = 1)"
    );
  }

  @Test
  public void while_statement() {
    verifyCfg(
      "before(succ = [cond_block], elem = 1)",
      "while cond_block(succ = [while_body, END], elem = 1):",
      "  while_body(succ = [cond_block], elem = 1)"
    );
  }

  @Test
  public void continue_statement() {
    verifyCfg(
      "before(succ = [cond_block], elem = 1)",
      "while cond_block(succ = [while_body, END], elem = 1):",
      "  while_body(succ = [cond_block], elem = 2, syntSucc = after_continue)",
      "  continue",
      "  after_continue(succ = [cond_block], elem = 1)"
    );
  }

  @Test
  public void continue_nested_while() {
    verifyCfg(
      "while cond_block(succ = [cond_block_inner, END]):",
      "  while cond_block_inner(succ = [cond_block, inner_while_block]):",
      "    inner_while_block(succ = [cond_block_inner], syntSucc = after_continue)",
      "    continue",
      "    after_continue(succ = [cond_block_inner])"
    );
  }

  @Test
  public void for_statement() {
    verifyCfg(
      "before(succ = [cond_block], elem = 2)",
      "for cond_block(succ = [for_body, END], elem = 1) in collection:",
      "  for_body(succ = [cond_block], elem = 1)"
    );
  }

  @Test
  public void continue_statement_in_for() {
    verifyCfg(
      "before(succ = [cond_block], elem = 2)",
      "for cond_block(succ = [for_body, END], elem = 1) in collection:",
      "  for_body(succ = [cond_block], elem = 2, syntSucc = after_continue)",
      "  continue",
      "  after_continue(succ = [cond_block], elem = 1)"
    );
  }

  @Test
  public void continue_nested_for() {
    verifyCfg(
      "before(succ = [cond_block])",
      "for cond_block(succ = [outer_for_block, END]) in collection1:",
      "  outer_for_block(succ = [cond_block_inner])",
      "  for cond_block_inner(succ = [inner_for_block, cond_block]) in collection2:",
      "    inner_for_block(succ = [cond_block_inner], syntSucc = after_continue)",
      "    continue",
      "    after_continue(succ = [cond_block_inner])"
    );
  }

  private ControlFlowGraph verifyCfg(String... lines) {
    ControlFlowGraph cfg = cfg(lines);
    CfgValidator.assertCfgStructure(cfg);
    assertNoEmptyBlocksInCFG(cfg);
    return cfg;
  }

  private void assertNoEmptyBlocksInCFG(ControlFlowGraph cfg) {
    cfg.blocks().stream()
      .filter(block -> !(block instanceof PythonCfgEndBlock))
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
