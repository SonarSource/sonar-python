/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.cfg.fixpoint;

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.cfg.CfgValidator;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.python.PythonTestUtils;
import org.sonar.plugins.python.api.symbols.Symbol;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * This Live Variable Analysis Test uses a meta-language to specify the expected LVA values for each basic block.
 * <p>
 * Convention:
 * <p>
 * 1. the metadata is specified as a function call with the form:
 * <p>
 * {@code block1( succ = [block2, END], liveIn = [x, y], liveOut = [y], gen = [x, y], kill = [x] ); }
 * where the arguments are assignments to:
 * - 'succ' is a bracketed array of expected successor ids. For branching blocks, the true successor must be first.
 * - 'liveIn'  - the live variables that enter the block
 * - 'liveOut' - the live variables that exit the block
 * - 'gen'     - the variables that are consumed by the block
 * - 'kill'    - the variables that are killed (overwritten) by the block
 * <p>
 * 2. each basic block must contain a function call with this structure as the first statement
 * - exception: a Label is before the block function call
 * <p>
 * 3. the name of the function is the identifier of the basic block
 */
class LiveVariablesAnalysisTest {
  private PythonFile file = Mockito.mock(PythonFile.class, "file1.py");

  @Test
  void test_simple_kill() {
    verifyLiveVariableAnalysis(
      "block( succ = [END], liveIn = [], liveOut = [], gen = [], kill = [_foo, _bar, _qix])",
      "foo = 1",
      "bar = baz()",
      "qix = 1 + 2");
  }

  @Test
  void test_simple_gen() {
    verifyLiveVariableAnalysisWithArgs(
      "foo, bar",
      "block( succ = [END], liveIn = [_foo, _bar], liveOut = [], gen = [_foo, _bar])",
      "f(foo, bar)");
  }

  @Test
  void test_complex_reads_and_writes() {
    verifyLiveVariableAnalysisWithArgs(
      "a,b",
      "block( succ = [END], liveIn = [_a,_b], liveOut = [], gen = [_a,_b], kill = [])",
      "a[b] = 1");

    // R, R
    verifyLiveVariableAnalysisWithArgs(
      "a",
      "block( succ = [END], liveIn = [_a], liveOut = [], gen = [_a], kill = [])",
      "read(a)",
      "read(a)");

    // RW, R
    verifyLiveVariableAnalysisWithArgs("a",
      "block( succ = [END], liveIn = [_a], liveOut = [], gen = [_a], kill = [_a])",
      "a = read(a)",
      "read(a)");

    // RW, W
    verifyLiveVariableAnalysisWithArgs("a",
      "block( succ = [END], liveIn = [_a], liveOut = [], gen = [_a], kill = [_a])",
      "a = read(a)",
      "a = 1");

    // R, W
    verifyLiveVariableAnalysisWithArgs("a",
      "block( succ = [END], liveIn = [_a], liveOut = [], gen = [_a], kill = [_a])",
      "read(a)",
      "a = 1");

    // W, R
    verifyLiveVariableAnalysis(
      "block( succ = [END], liveIn = [], liveOut = [], gen = [], kill = [_a])",
      "a = 1",
      "read(a)");

    // W, W
    verifyLiveVariableAnalysis(
      "block( succ = [END], liveIn = [], liveOut = [], gen = [], kill = [_a])",
      "a = 1",
      "a = 1");

    // R, W, R
    verifyLiveVariableAnalysisWithArgs("a",
      "block( succ = [END], liveIn = [_a], liveOut = [], gen = [_a], kill = [_a])",
      "read(a)",
      "a = 1",
      "read(a)");

    // R, RW, R
    verifyLiveVariableAnalysisWithArgs("a",
      "block( succ = [END], liveIn = [_a], liveOut = [], gen = [_a], kill = [_a])",
      "read(a)",
      "a = read(a)",
      "read(a)");

    // W, R, W
    verifyLiveVariableAnalysis(
      "block( succ = [END], liveIn = [], liveOut = [], gen = [], kill = [_a])",
      "a = 1",
      "read(a)",
      "a = 1");
  }

  @Test
  void test_write_before_read() {
    verifyLiveVariableAnalysis("" +
      "condition( succ = [body, END], liveIn = [], liveOut = [], gen = [], kill = [_a])",
      "a = 1",
      "foo(a)",
      "if p:",
      "  body( succ = [END], liveIn = [], liveOut = [], gen = [], kill = [_a]);",
      "  a = 1",
      "  foo(a)");
  }

  @Test
  void test_while() {
    verifyLiveVariableAnalysis(
      "before_loop( succ = [cond], liveIn = [], liveOut = [_a], gen = [], kill = [_a])",
      "a = 1",
      "while cond( succ = [after_loop, if_cond], liveIn = [_a], liveOut = [_a], gen = [], kill = []):",
      "  if if_cond(succ = [cond, body], liveIn = [_a], liveOut = [_a], gen = [], kill = []):",
      "    body( succ = [cond], liveIn = [], liveOut = [_a], gen = [], kill = [_a]);",
      "    a = 1",
      "    foo(a)",
      "after_loop(succ = [END], liveIn = [_a], liveOut = [], gen = [_a], kill = [])",
      "foo(a)");
  }

  @Test
  void test_foreach() {
    verifyLiveVariableAnalysis(
      "before_loop( succ = [cond], liveIn = [], liveOut = [_a], gen = [], kill = [_a])",
      "a = 1",
      "for cond( succ = [after_loop, if_cond], liveIn = [_a], liveOut = [_a], gen = [], kill = []) in elems:",
      "  if if_cond(succ = [cond, body], liveIn = [_a], liveOut = [_a], gen = [], kill = []):",
      "    body( succ = [cond], liveIn = [], liveOut = [_a], gen = [], kill = [_a]);",
      "    a = 1",
      "    foo(a)",
      "after_loop(succ = [END], liveIn = [_a], liveOut = [], gen = [_a], kill = [])",
      "foo(a)");
  }

  @Test
  void read_symbols() {
    List<String> lines = Arrays.asList(
      "foo = 1",
      "bar = f()",
      "bar += f()",
      "read(bar)",
      "qix += 1 + 2"
    );
    FileInput fileInput = PythonTestUtils.parse("def wrapper():", lines.stream().map(s -> "  " + s).collect(Collectors.joining("\n")));
    FunctionDef fun = (FunctionDef) fileInput.statements().statements().get(0);
    ControlFlowGraph cfg = ControlFlowGraph.build(fun, file);
    LiveVariablesAnalysis analysis = LiveVariablesAnalysis.analyze(cfg);
    Set<Symbol> readSymbols = analysis.getReadSymbols();
    assertThat(readSymbols).extracting("name").containsExactlyInAnyOrder("bar", "qix");
  }

  @Test
  void is_symbol_used_in_block() {
    List<String> lines = Arrays.asList(
      "a = 10",
      "b = 42",
      "print(a+b)"
    );
    FileInput fileInput = PythonTestUtils.parse("def wrapper():", lines.stream().map(s -> "  " + s).collect(Collectors.joining("\n")));
    FunctionDef fun = (FunctionDef) fileInput.statements().statements().get(0);
    ControlFlowGraph cfg = ControlFlowGraph.build(fun, file);
    LiveVariablesAnalysis analysis = LiveVariablesAnalysis.analyze(cfg);
    fun.localVariables().forEach(symbol -> assertThat(analysis.getLiveVariables(cfg.start()).isSymbolUsedInBlock(symbol)).isTrue());
  }


  private void verifyLiveVariableAnalysis(String... lines) {
    verifyLiveVariableAnalysisWithArgs("", lines);
  }

  private void verifyLiveVariableAnalysisWithArgs(String argList, String... lines) {
    FileInput fileInput = PythonTestUtils.parse("def wrapper(" + argList + "):", Arrays.stream(lines).map(s -> "  " + s).collect(Collectors.joining("\n")));
    FunctionDef fun = (FunctionDef) fileInput.statements().statements().get(0);
    ControlFlowGraph cfg = ControlFlowGraph.build(fun, file);
    LiveVariablesAnalysis analysis = LiveVariablesAnalysis.analyze(cfg);
    CfgValidator.assertLiveVariables(cfg, analysis);
  }

}
