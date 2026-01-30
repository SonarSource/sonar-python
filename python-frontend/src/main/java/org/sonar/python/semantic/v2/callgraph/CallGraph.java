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

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Tree;

public class CallGraph {
  public static final CallGraph EMPTY = new CallGraph(Map.of(), Map.of());

  private final Map<String, List<CallGraphNode>> usageEdges;
  private final Map<String, List<CallGraphNode>> callEdges;

  private CallGraph(Map<String, List<CallGraphNode>> usageEdges, Map<String, List<CallGraphNode>> callEdges) {
    this.usageEdges = usageEdges;
    this.callEdges = callEdges;
  }

  public List<CallGraphNode> getUsages(String fqn) {
    return Collections.unmodifiableList(usageEdges.getOrDefault(fqn, List.of()));
  }

  public List<CallGraphNode> getCalls(String fqn) {
    return Collections.unmodifiableList(callEdges.getOrDefault(fqn, List.of()));
  }

  public Iterable<CallGraphNode> forward(String startFqn) {
    return new CallGraphIterable(startFqn, this, CallGraph::getCalls);
  }

  public Stream<CallGraphNode> forwardStream(String startFqn) {
    return StreamSupport.stream(forward(startFqn).spliterator(), false);
  }

  public Iterable<CallGraphNode> backward(String startFqn) {
    return new CallGraphIterable(startFqn, this, CallGraph::getUsages);
  }

  public Stream<CallGraphNode> backwardStream(String startFqn) {
    return StreamSupport.stream(backward(startFqn).spliterator(), false);
  }

  public static class Builder {
    private Map<String, CallGraphNode> cache = new HashMap<>();
    private Map<String, List<CallGraphNode>> builderUsageEdges = new HashMap<>();
    private Map<String, List<CallGraphNode>> builderCallEdges = new HashMap<>();

    /**
     * Adds a usage edge from one function to another in the call graph.
     * @param from The function that is calling the function
     * @param to The functions that is being called
     * @param callingFunction The function that is calling the function
     * @param calledFunction The function that is being called
     * @return this builder
     */
    public Builder addUsage(String from, String to, Tree callingFunction, @Nullable Tree calledFunction) {
      var fromNode = getOrCreateNode(from, callingFunction);
      var toNode = getOrCreateNode(to, calledFunction);
      builderUsageEdges.computeIfAbsent(to, k -> new ArrayList<>()).add(fromNode);
      builderCallEdges.computeIfAbsent(from, k -> new ArrayList<>()).add(toNode);
      return this;
    }

    private CallGraphNode getOrCreateNode(String fqn, @Nullable Tree tree) {
      return cache.computeIfAbsent(fqn, k -> new CallGraphNode(fqn, new WeakReference<>(tree)));
    }

    public CallGraph build() {
      var graph = new CallGraph(builderUsageEdges, builderCallEdges);
      builderUsageEdges = new HashMap<>();
      builderCallEdges = new HashMap<>();
      cache = new HashMap<>();
      return graph;
    }

  }
}
