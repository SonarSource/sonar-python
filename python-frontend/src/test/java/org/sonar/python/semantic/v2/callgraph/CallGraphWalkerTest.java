/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.semantic.v2.callgraph;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.TriBool;

import static org.assertj.core.api.Assertions.assertThat;

class CallGraphWalkerTest {

  @Test
  void should_return_true_when_predicate_matches() {
    CallGraph callGraph = new CallGraph.Builder()
      .addUsage("func1", "start")
      .addUsage("func2", "func1")
      .build();

    Predicate<CallGraphNode> predicate = node -> "func2".equals(node.fqn());

    CallGraphWalker walker = new CallGraphWalker(callGraph);
    TriBool result = walker.isUsedFrom("start", predicate);

    assertThat(result).isEqualTo(TriBool.TRUE);
  }

  @Test
  void should_return_false_when_no_match_is_found() {
    CallGraph callGraph = new CallGraph.Builder()
      .addUsage("func1", "start")
      .addUsage("func2", "func1")
      .build();

    Predicate<CallGraphNode> predicate = node -> "nonExistentFunc".equals(node.fqn());

    CallGraphWalker walker = new CallGraphWalker(callGraph);
    TriBool result = walker.isUsedFrom("start", predicate);

    assertThat(result).isEqualTo(TriBool.FALSE);
  }

  @Test
  void should_not_test_start_node() {
    CallGraph callGraph = new CallGraph.Builder()
      .addUsage("func1", "start")
      .build();

    Predicate<CallGraphNode> predicate = node -> "start".equals(node.fqn());

    CallGraphWalker walker = new CallGraphWalker(callGraph);
    TriBool result = walker.isUsedFrom("start", predicate);

    assertThat(result).isEqualTo(TriBool.FALSE);
  }

  @Test
  void should_return_unknown_when_max_visited_is_reached() {
    CallGraph callGraph = new CallGraph.Builder()
      .addUsage("func1", "start")
      .addUsage("func2", "func1")
      .addUsage("func3", "func2")
      .addUsage("func1", "func3")
      .build();

    Predicate<CallGraphNode> predicate = node -> "nonExistentFunc".equals(node.fqn());

    CallGraphWalker walker = new CallGraphWalker(callGraph, 2);
    TriBool result = walker.isUsedFrom("start", predicate);

    assertThat(result).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void should_handle_cycles_in_call_graph() {
    CallGraph callGraph = new CallGraph.Builder()
      .addUsage("start", "func1")
      .addUsage("func1", "func2")
      .addUsage("func2", "start")
      .build();

    Predicate<CallGraphNode> predicate = node -> "nonExistentFunc".equals(node.fqn());

    CallGraphWalker walker = new CallGraphWalker(callGraph);
    TriBool result = walker.isUsedFrom("start", predicate);

    assertThat(result).isEqualTo(TriBool.FALSE);
  }

  @Test
  void should_visit_breath_first() {
    CallGraph callGraph = new CallGraph.Builder()
      .addUsage("func1", "start")
      .addUsage("func2", "start")
      .addUsage("func3", "func1")
      .build();

    List<String> visitedNodes = new ArrayList<>();

    Predicate<CallGraphNode> predicate = node -> {
      visitedNodes.add(node.fqn());
      return false;
    };

    CallGraphWalker walker = new CallGraphWalker(callGraph);
    TriBool result = walker.isUsedFrom("start", predicate);

    assertThat(result).isEqualTo(TriBool.FALSE);
    assertThat(visitedNodes).containsExactly("func1", "func2", "func3");
  }

  @Test
  void should_handle_empty_usages() {
    CallGraph callGraph = CallGraph.EMPTY;

    Predicate<CallGraphNode> predicate = node -> "anything".equals(node.fqn());

    CallGraphWalker walker = new CallGraphWalker(callGraph);
    TriBool result = walker.isUsedFrom("start", predicate);

    assertThat(result).isEqualTo(TriBool.FALSE);
  }
}
