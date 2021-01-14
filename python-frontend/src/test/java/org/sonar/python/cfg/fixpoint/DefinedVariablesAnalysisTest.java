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
package org.sonar.python.cfg.fixpoint;

import java.util.Arrays;
import java.util.stream.Collectors;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.cfg.CfgValidator;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.python.PythonTestUtils;

/**
 * This Defined Variable Analysis Test uses a meta-language to specify the expected values for each basic block.
 * <p>
 * Convention:
 * <p>
 * 1. the metadata is specified as a function call with the form:
 * <p>
 * {@code block1( succ = [block2, END], defIn = [x, y], defOut = [y], gen = [x, y], kill = [x] ); }
 * where the arguments are assignments to:
 * - 'succ' is a bracketed array of expected successor ids. For branching blocks, the true successor must be first.
 * - 'defIn'  - the defined variables that enter the block
 * - 'defOut' - the defined variables that exit the block
 * - 'gen'     - the variables that are consumed by the block
 * - 'kill'    - the variables that are killed (overwritten) by the block
 * <p>
 * 2. each basic block must contain a function call with this structure as the first statement
 * - exception: a Label is before the block function call
 * <p>
 * 3. the name of the function is the identifier of the basic block
 */
public class DefinedVariablesAnalysisTest {
  private PythonFile file = Mockito.mock(PythonFile.class, "file1.py");

  @Test
  public void test_simple_def() {
    verifyDefVariableAnalysis(
      "block( succ = [END], defIn= [], defOut = [_foo, _bar, _qix])",
      "foo = 1",
      "bar = baz()",
      "qix = 1 + 2");
  }

  @Test
  public void test_fn() {
    verifyDefVariableAnalysis(
      "block( succ = [END], defIn= [], defOut = [_foo, _fn])",
      "foo = 1",
      "def fn(): print(foo)");
  }

  @Test
  public void test_not_def() {
    verifyDefVariableAnalysis(
      "block( succ = [END], defOut = [], defIn = [])",
      "read(a)",
      "read(a)");
  }

  @Test
  public void test_write_before_read() {
    verifyDefVariableAnalysis(
      "condition( succ = [body, END], defIn = [], defOut = [_a])",
      "a = 1",
      "foo(a)",
      "if p:",
      "  body( succ = [END], defIn = [_a], defOut = [_a])",
      "  foo(a)");
  }

  @Test
  public void test_while() {
    verifyDefVariableAnalysis(
      "before_loop( succ = [cond], defIn = [], defOut = [_a])",
      "a = 1",
      "while cond( succ = [after_loop, if_cond], defIn = [_a], defOut = [_a], gen = [], kill = []):",
      "  if if_cond(succ = [cond, body], defIn = [_a], defOut = [_a], gen = [], kill = []):",
      "    body( succ = [cond], defIn = [_a], defOut = [_a], gen = [], kill = [_a]);",
      "    a = 1",
      "    foo(a)",
      "after_loop(succ = [END], defIn = [_a], defOut = [_a], gen = [_a], kill = [])",
      "foo(a)");
  }
  private void verifyDefVariableAnalysis(String... lines) {
    FileInput fileInput = PythonTestUtils.parse("def wrapper():", Arrays.stream(lines).map(s -> "  " + s).collect(Collectors.joining("\n")));
    FunctionDef fun = (FunctionDef) fileInput.statements().statements().get(0);
    ControlFlowGraph cfg = ControlFlowGraph.build(fun, file);
    DefinedVariablesAnalysis analysis = DefinedVariablesAnalysis.analyze(cfg, fun.localVariables());
    CfgValidator.assertDefinedVariables(cfg, analysis);
  }

}
