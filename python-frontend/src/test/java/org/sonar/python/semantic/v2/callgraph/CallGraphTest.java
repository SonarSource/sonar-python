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

import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;
import static org.mockito.Mockito.mock;

class CallGraphTest {
  @Test
  void testNull() {
    CallGraph graph = new CallGraph.Builder()
      .addUsage("function1", "function2", null, null)
      .build();

    List<CallGraphNode> calls = graph.getCalls("function1");
    assertThat(calls)
      .extracting(CallGraphNode::fqn, CallGraphNode::tree)
      .containsExactly(tuple("function2", Optional.empty()));
  }

  @Test
  void testGetUsages() {
    CallGraph graph = new CallGraph.Builder()
      .addUsage("function2", "function1", null, null)
      .addUsage("function3", "function1", null, null)
      .build();

    List<CallGraphNode> usagesForNonExistantFunction = graph.getUsages("some_non_existent_function");
    assertThat(usagesForNonExistantFunction).isEmpty();

    List<CallGraphNode> usagesOfFunction1 = graph.getUsages("function1");
    assertThat(usagesOfFunction1)
      .extracting(CallGraphNode::fqn)
      .containsExactly("function2", "function3");
  }

  @Test
  void testForward() {
    Tree fun1 = mock();
    Tree fun2 = mock();
    Tree fun3 = mock();
    CallGraph graph = new CallGraph.Builder()
      .addUsage("function1", "function2", fun1, fun2)
      .addUsage("function1", "function3", fun1, fun3)
      .build();

    List<CallGraphNode> forward = graph.forwardStream("function1").toList();
    assertThat(forward)
      .extracting(CallGraphNode::fqn, CallGraphNode::tree)
      .containsExactly(
        tuple("function2", Optional.of(fun2)),
        tuple("function3", Optional.of(fun3)));
  }

  @Test
  void testForwardCycle() {
    Tree fun1 = mock();
    Tree fun2 = mock();
    Tree fun3 = mock();
    CallGraph graph = new CallGraph.Builder()
      .addUsage("function3", "function1", fun3, fun1)
      .addUsage("function1", "function2", fun1, fun2)
      .addUsage("function2", "function3", fun1, fun3)
      .build();

    List<CallGraphNode> forward = graph.forwardStream("function2").toList();
    assertThat(forward)
      .extracting(CallGraphNode::fqn, CallGraphNode::tree)
      .containsExactly(
        tuple("function3", Optional.of(fun3)),
        tuple("function1", Optional.of(fun1)),
        tuple("function2", Optional.of(fun2)));
  }

  @Test
  void testBackward() {
    Tree fun1 = mock();
    Tree fun2 = mock();
    Tree fun3 = mock();
    CallGraph graph = new CallGraph.Builder()
      .addUsage("function2", "function1", fun2, fun1)
      .addUsage("function3", "function1", fun3, fun1)
      .build();

    List<CallGraphNode> backward = graph.backwardStream("function1").toList();
    assertThat(backward)
      .extracting(CallGraphNode::fqn, CallGraphNode::tree)
      .containsExactly(
        tuple("function2", Optional.of(fun2)),
        tuple("function3", Optional.of(fun3)));
  }

  @Test
  void testBackwardCycle() {
    Tree fun1 = mock();
    Tree fun2 = mock();
    Tree fun3 = mock();
    CallGraph graph = new CallGraph.Builder()
      .addUsage("function2", "function1", fun2, fun1)
      .addUsage("function3", "function1", fun3, fun1)
      .addUsage("function1", "function2", fun1, fun2)
      .build();

    List<CallGraphNode> backward = graph.backwardStream("function1").toList();
    assertThat(backward)
      .extracting(CallGraphNode::fqn, CallGraphNode::tree)
      .containsExactly(
        tuple("function2", Optional.of(fun2)),
        tuple("function3", Optional.of(fun3)),
        tuple("function1", Optional.of(fun1)));
  }
}
