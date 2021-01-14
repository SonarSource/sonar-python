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

import java.util.Arrays;
import java.util.stream.Collectors;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.cfg.PythonCfgBlock;
import org.sonar.python.cfg.PythonCfgBranchingBlock;
import org.sonar.python.cfg.PythonCfgEndBlock;
import org.sonar.python.cfg.PythonCfgSimpleBlock;

import static org.assertj.core.api.Assertions.assertThat;

public class ControlFlowGraphTest {

  private PythonFile file = Mockito.mock(PythonFile.class, "file1.py");

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
      "elif before_elif_body_1(succ = [elif_body_1, else_body], elem = 1):",
      "  elif_body_1(succ = [END], elem = 1)",
      "else:",
      "  else_body(succ = [END], elem = 1)"
    );
  }

  @Test
  public void while_statement() {
    ControlFlowGraph cfg = verifyCfg(
      "before(succ = [cond_block], elem = 1)",
      "while cond_block(succ = [while_body, END], elem = 1):",
      "  while_body(succ = [cond_block], elem = 1)"
    );
    CfgBranchingBlock condBlock = (CfgBranchingBlock) cfg.start().successors().iterator().next();
    assertThat(condBlock.branchingTree().getKind()).isEqualTo(Kind.WHILE_STMT);
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
  public void continue_outside_loop() {
    assertThat(cfg("continue")).isNull();
  }

  @Test
  public void return_outside_function() {
    assertThat(ControlFlowGraph.build(PythonTestUtils.parse("return"), file)).isNull();
    assertThat(cfg(
      "class Foo:",
      "  return"
      )).isNull();
  }

  @Test
  public void continue_nested_while() {
    verifyCfg(
      "while cond_block(succ = [cond_block_inner, END]):",
      "  while cond_block_inner(succ = [inner_while_block, cond_block]):",
      "    inner_while_block(succ = [cond_block_inner], syntSucc = after_continue)",
      "    continue",
      "    after_continue(succ = [cond_block_inner])"
    );
  }

  @Test
  public void break_statement() {
    verifyCfg(
      "while cond(succ = [while_body, after_while], elem = 1):",
      "  while_body(succ = [if_body, after_break], elem = 2)",
      "  if cond2:",
      "    if_body(succ = [after_while], elem = 2, syntSucc = after_break)",
      "    break",
      "  after_break(succ = [cond], elem = 1)",
      "after_while(succ = [END], elem = 1)"
    );
  }

  @Test
  public void for_statement() {
    verifyCfg(
      "before(succ = [cond_block], elem = 2)",
      "for cond_block(succ = [for_body, END], elem = 2) in collection:",
      "  for_body(succ = [cond_block], elem = 1)"
    );
  }

  @Test
  public void for_statement_else() {
    verifyCfg(
      "before(succ = [cond_block], elem = 2)",
      "for cond_block(succ = [for_body, else_body], elem = 2) in collection:",
      "  for_body(succ = [cond_block], elem = 1)",
      "else:",
      "  else_body(succ = [END], elem = 1)"
    );
  }

  @Test
  public void for_statement_else_break() {
    verifyCfg(
      "before(succ = [cond_block])",
      "for cond_block(succ = [for_body, else_body]) in collection:",
      "  for_body(succ = [after_for], syntSucc = cond_block)",
      "  break",
      "else:",
      "  else_body(succ = [after_for])",
      "after_for(succ = [END])"
    );
  }

  @Test
  public void continue_statement_in_for() {
    verifyCfg(
      "before(succ = [cond_block], elem = 2)",
      "for cond_block(succ = [for_body, END], elem = 2) in collection:",
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

  @Test
  public void simple_try_except() {
    verifyCfg(
      "before(succ = [try_block])",
      "try:",
      "  try_block(succ = [after, except_cond])",
      "except except_cond(succ = [except_block, END]) as e:",
      "  except_block(succ = [after])",
      "after(succ = [END])"
    );
  }

  @Test
  public void simple_try_except_finally() {
    verifyCfg(
      "before(succ = [try_block])",
      "try:",
      "  try_block(succ = [finally_block, except_cond])",
      "except except_cond(succ = [except_block, finally_block]) as e:",
      "  except_block(succ = [finally_block])",
      "finally:",
      "  finally_block(succ = [after, END])",
      "after(succ=[END])"
    );
  }

  @Test
  public void simple_try_except_finally_else() {
    verifyCfg(
      "before(succ = [try_block])",
      "try:",
      "  try_block(succ = [else_block, except_cond_1])",
      "except except_cond_1(succ = [except_block_1, except_cond_2]) as e:",
      "  except_block_1(succ = [finally_block])",
      "except except_cond_2(succ = [except_block_2, finally_block]) as e:",
      "  except_block_2(succ = [finally_block])",
      "else:",
      "  else_block(succ = [finally_block])",
      "finally:",
      "  finally_block(succ = [END, END])"
    );
  }

  @Test
  public void simple_try_except_value_error() {
    ControlFlowGraph cfg = cfg(
      "try:",
      "  foo()",
      "except ValueError as error:",
      "  print(error)"
    );
    assertNoEmptyBlocksInCFG(cfg);
    CfgBlock start = cfg.start();
    assertThat(start.elements()).extracting(Tree::getKind).containsExactly(Kind.EXPRESSION_STMT);
    assertThat(start.successors()).hasSize(2);
    CfgBlock exceptCondition = start.successors().stream().filter(succ -> !(succ instanceof PythonCfgEndBlock)).findFirst().get();
    assertThat(exceptCondition.elements()).extracting(Tree::getKind).containsExactly(Kind.NAME, Kind.NAME);
    assertThat(exceptCondition.successors()).hasSize(2);
  }

  @Test
  public void simple_try_except_all_exceptions() {
    ControlFlowGraph cfg = cfg(
      "try:",
      "  foo()",
      "except:",
      "  print('exception')"
    );
    assertNoEmptyBlocksInCFG(cfg);
    CfgBlock start = cfg.start();
    assertThat(start.elements()).extracting(Tree::getKind).containsExactly(Kind.EXPRESSION_STMT);
    assertThat(start.successors()).hasSize(2);
    CfgBlock exceptCondition = start.successors().stream().filter(succ -> !(succ instanceof PythonCfgEndBlock)).findFirst().get();
    assertThat(exceptCondition.elements()).isEmpty();
    assertThat(exceptCondition.successors()).hasSize(2);
  }

  //TODO nested try

  @Test
  public void raise_without_try() {
    verifyCfg(
      "before(succ = [END], elem = 2, syntSucc = after)",
      "raise my_error",
      "after(succ = [END], elem = 1)");
  }

  @Test
  public void raise_in_try() {
    verifyCfg(
      "before_try(succ = [try_body], elem = 1)",
      "try:",
      "  try_body(succ = [cond1], syntSucc = after_raise, elem = 2)",
      "  raise my_error",
      "  after_raise(succ = [after_try, cond1], elem = 1)",
      "except cond1(succ = [except_body, END], elem = 1):",
      "  except_body(succ = [after_try], elem = 1)",
      "after_try(succ = [END], elem = 1)");
  }

  @Test
  public void return_in_try() {
    ControlFlowGraph cfg = cfg(
      "before_try(succ = [try_body])",
      "try:",
      "  try_body(succ = [except_cond], syntSucc = _empty)",
      "  return",
      "except except_cond(succ = [except_block, finally_block]):",
      " except_block(succ = [finally_block])",
      "finally:",
      "  finally_block(succ = [after_try, END])",
      "  pass",
      "after_try(succ = [END])");

    ExpectedCfgStructure expectedCfgStructure = ExpectedCfgStructure.parse(cfg.blocks(), expected -> {
      expected.createEmptyBlockExpectation()
        .withSuccessorsIds("except_cond", "finally_block");
      return expected;
    });
    new CfgValidator(expectedCfgStructure).assertCfg(cfg);
  }

  @Test
  public void break_in_try() {
    verifyCfg(
      "before_while(succ = [cond])",
      "while cond(succ = [try_body, after_while]):",
      "  try:",
      "    try_body(succ = [finally_block], syntSucc = finally_block)",
      "    break",
      "  finally:",
      "    finally_block(succ = [after_try, END])",
      "    pass",
      "  after_try(succ = [cond])",
      "after_while(succ = [END])");
  }

  @Test
  public void continue_in_try() {
    ControlFlowGraph cfg = cfg(
      "before_while(succ = [cond])",
      "while cond(succ = [try_body, after_while]):",
      "  try:",
      "    try_body(succ = [except_cond], syntSucc = _empty)",
      "    continue",
      "  except except_cond(succ = [except_block, finally_block]):",
      "    except_block(succ = [finally_block])",
      "    print()",
      "  finally:",
      "    finally_block(succ = [after_try, END])",
      "    pass",
      "  after_try(succ = [cond])",
      "after_while(succ = [END])");
    ExpectedCfgStructure expectedCfgStructure = ExpectedCfgStructure.parse(cfg.blocks(), expected -> {
      expected.createEmptyBlockExpectation()
        .withSuccessorsIds("except_cond", "finally_block");
      return expected;
    });
    new CfgValidator(expectedCfgStructure).assertCfg(cfg);
  }

  @Test
  public void return_in_except() {
    verifyCfg(
      "before_try(succ = [try_body])",
      "try:",
      "  try_body(succ = [finally_block, except_cond])",
      "except except_cond(succ = [except_block, finally_block]):",
      "  except_block(succ = [finally_block], syntSucc = finally_block)",
      "  return",
      "finally:",
      "  finally_block(succ = [after_try, END])",
      "  pass",
      "after_try(succ = [END])");
  }

  @Test
  public void return_in_else() {
    verifyCfg(
      "before_try(succ = [try_body])",
      "try:",
      "  try_body(succ = [else_block, except_cond])",
      "except except_cond(succ = [except_block, finally_block]):",
      "  except_block(succ = [finally_block])",
      "else:",
      "  else_block(succ = [finally_block], syntSucc = finally_block)",
      "  return",
      "finally:",
      "  finally_block(succ = [after_try, END])",
      "  pass",
      "after_try(succ = [END])");
  }


  @Test
  public void continue_in_except() {
    verifyCfg(
      "before_while(succ = [cond])",
      "while cond(succ = [try_body, after_while]):",
      "  try:",
      "    try_body(succ = [finally_block, except_cond])",
      "  except except_cond(succ = [except_block, finally_block]):",
      "    except_block(succ = [finally_block], syntSucc=finally_block)",
      "    continue",
      "  finally:",
      "    finally_block(succ = [after_try, END])",
      "  after_try(succ = [cond])",
      "after_while(succ = [END])");
  }

  @Test
  public void continue_in_else() {
    verifyCfg(
      "before_while(succ = [cond])",
      "while cond(succ = [try_body, after_while]):",
      "  try:",
      "    try_body(succ = [else_block, except_cond])",
      "  except except_cond(succ = [except_block, finally_block]):",
      "    except_block(succ = [finally_block])",
      "  else:",
      "    else_block(succ = [finally_block], syntSucc=finally_block)",
      "    continue",
      "  finally:",
      "    finally_block(succ = [after_try, END])",
      "  after_try(succ = [cond])",
      "after_while(succ = [END])");
  }

  @Test
  public void break_in_except() {
    verifyCfg(
      "before_while(succ = [cond])",
      "while cond(succ = [try_body, after_while]):",
      "  try:",
      "    try_body(succ = [finally_block, except_cond])",
      "  except except_cond(succ = [except_block, finally_block]):",
      "    except_block(succ = [finally_block], syntSucc=finally_block)",
      "    break",
      "  finally:",
      "    finally_block(succ = [after_try, END])",
      "  after_try(succ = [cond])",
      "after_while(succ = [END])");
  }

  @Test
  public void break_in_else() {
    verifyCfg(
      "before_while(succ = [cond])",
      "while cond(succ = [try_body, after_while]):",
      "  try:",
      "    try_body(succ = [else_block, except_cond])",
      "  except except_cond(succ = [except_block, finally_block]):",
      "    except_block(succ = [finally_block])",
      "  else:",
      "    else_block(succ = [finally_block], syntSucc=finally_block)",
      "    break",
      "  finally:",
      "    finally_block(succ = [after_try, END])",
      "  after_try(succ = [cond])",
      "after_while(succ = [END])");
  }

  @Test
  public void with_statement() {
    verifyCfg(
      "before(succ = [with_block, END])",
      "with A() as a:",
      "  if with_block(succ = [if_body, END]):",
      "    if_body(succ = [END])"
    );
  }

  @Test
  public void CFGBlock_toString() {
    PythonCfgEndBlock endBlock = new PythonCfgEndBlock();
    assertThat(endBlock.toString()).isEqualTo("END");
    PythonCfgBlock pythonCfgBlock = new PythonCfgSimpleBlock(endBlock);
    assertThat(pythonCfgBlock.toString()).isEqualTo("empty");
    ControlFlowGraph cfg = cfg(
     "pass",
     "assert 2"
    );
    assertThat(cfg.start().toString()).isEqualTo("2:2:PASS_STMT;ASSERT_STMT");
  }

  @Test
  public void CFG_toString() {
    ControlFlowGraph cfg = cfg("" +
      "if x:",
      "    return 1",
      "else:",
      "    foo()",
      "    return 2"
    );
    assertThat(cfg.toString()).isEqualTo("" +
      "0[label=\"2:2:NAME\"];" +
      "1[label=\"3:6:RETURN_STMT\"];" +
      "2[label=\"5:6:EXPRESSION_STMT;RETURN_STMT\"];" +
      "3[label=\"END\"];" +
      "0->1;0->2;1->3;1->3[style=dotted];2->3;2->3[style=dotted];");
  }

  @Test
  public void class_def() {
    verifyCfg(
      "before(succ = [if_body, END], elem = 3)",
      "class A:",
      "  if cond:",
      "    if_body(succ = [END], elem = 1)"
    );
  }

  /**
   * Because the predecessors are constructed based on the successors, there is no need to have assertions on predecessors on all other CFG tests
   */
  @Test
  public void if_stmt_test_predecessors() {
    verifyCfg("" +
      "before(succ = [if_body, after_if], pred = [])",
      "foo()",
      "if a: ",
      "  if_body(succ = [after_if], pred = [before])",
      "after_if(succ = [END], pred = [before, if_body])");
  }

  @Test
  public void parameters() {
    FileInput fileInput = PythonTestUtils.parse("def f(p1, p2): pass");
    FunctionDef fun = (FunctionDef) fileInput.statements().statements().get(0);
    ControlFlowGraph cfg = ControlFlowGraph.build(fun, file);
    assertThat(cfg.start().elements()).extracting(element -> ((Parameter) element).name().name()).containsExactlyInAnyOrder("p1", "p2");

    fileInput = PythonTestUtils.parse("def f((p1, (p2, p3)), p4): pass");
    fun = (FunctionDef) fileInput.statements().statements().get(0);
    cfg = ControlFlowGraph.build(fun, file);
    assertThat(cfg.start().elements()).extracting(element -> ((Parameter) element).name().name()).containsExactlyInAnyOrder("p1", "p2", "p3", "p4");
  }

  @Test
  public void successors_predecessors_order() {
    ControlFlowGraph cfg = cfg(
      "if p:",
      "  print('True')"
    );
    PythonCfgBranchingBlock branchingBlock = (PythonCfgBranchingBlock) cfg.start();
    assertThat(branchingBlock.successors()).containsExactly(branchingBlock.trueSuccessor(), branchingBlock.falseSuccessor());
    assertThat(cfg.end().predecessors()).containsExactly(branchingBlock.trueSuccessor(), branchingBlock);
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
    FunctionDef fun = (FunctionDef) fileInput.statements().statements().get(0);
    return ControlFlowGraph.build(fun, file);
  }

  private ControlFlowGraph fileCfg(String... lines) {
    FileInput fileInput = PythonTestUtils.parse(lines);
    return ControlFlowGraph.build(fileInput, file);
  }
}
