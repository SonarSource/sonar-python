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

import java.util.Iterator;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.junit.Assert.assertThrows;

class CallGraphIterableTest {
  @Test
  void testException() {
    CallGraph graph = new CallGraph.Builder()
      .build();
    CallGraphIterable iterable = new CallGraphIterable("function1", graph, CallGraph::getUsages);

    Iterator<CallGraphNode> iterator = iterable.iterator();
    assertThatExceptionOfType(NoSuchElementException.class).isThrownBy(iterator::next);
  }

  @Test
  void testVisitedBehavior() {
    CallGraph graph = new CallGraph.Builder()
      .addUsage("function1", "function2", null, null)
      .addUsage("function2", "function1", null, null)
      .build();
    CallGraphIterable iterable = new CallGraphIterable("function1", graph, CallGraph::getUsages);

    Iterator<CallGraphNode> iterator = iterable.iterator();
    assertThat(iterator.next()).isEqualTo(new CallGraphNode("function2"));
    assertThat(iterator.next()).isEqualTo(new CallGraphNode("function1"));
    assertThrows(NoSuchElementException.class, iterator::next);
  }
}
