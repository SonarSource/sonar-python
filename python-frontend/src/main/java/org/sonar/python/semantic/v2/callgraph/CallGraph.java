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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CallGraph {
  public static final CallGraph EMPTY = new CallGraph(Map.of());

  private final Map<String, List<CallGraphNode>> usageEdges;

  private CallGraph(Map<String, List<CallGraphNode>> usageEdges) {
    this.usageEdges = usageEdges;
  }

  public List<CallGraphNode> getUsages(String fqn) {
    return Collections.unmodifiableList(usageEdges.getOrDefault(fqn, List.of()));
  }

  public static class Builder {
    private Map<String, List<CallGraphNode>> builderUsageEdges = new HashMap<>();

    /**
     * Adds a usage edge from one function to another in the call graph.
     * @param from The calling function's fully qualified name (FQN).
     * @param to The functions that is being called
     * @return this builder
     */
    public Builder addUsage(String from, String to) {
      var node = new CallGraphNode(from);
      builderUsageEdges.computeIfAbsent(to, k -> new ArrayList<>()).add(node);
      return this;
    }

    public CallGraph build() {
      var graph = new CallGraph(builderUsageEdges);
      builderUsageEdges = new HashMap<>();
      return graph;
    }
  }

}
