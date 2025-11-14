/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

class CallGraphTest {
  @Test
  void testGetUsages() {
    CallGraph graph = new CallGraph.Builder()
      .addUsage("function2", "function1")
      .addUsage("function3", "function1")
      .build();

    List<CallGraphNode> usagesForNonExistantFunction = graph.getUsages("some_non_existent_function");
    assertThat(usagesForNonExistantFunction).isEmpty();

    List<CallGraphNode> usagesOfFunction1 = graph.getUsages("function1");
    assertThat(usagesOfFunction1).containsExactly(new CallGraphNode("function2"), new CallGraphNode("function3"));
  }
}
