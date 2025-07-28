/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.TriBool;

public class CallGraphWalker {
  private static final int DEFAULT_MAX_VISITED = 30;

  private final int maxVisited;
  private final CallGraph callGraph;

  public CallGraphWalker(CallGraph callGraph, int maxVisited) {
    this.callGraph = callGraph;
    this.maxVisited = maxVisited;
  }
  
  public CallGraphWalker(CallGraph callGraph) {
    this(callGraph, DEFAULT_MAX_VISITED);
  }

  public TriBool isUsedFrom(String startFqn, Predicate<CallGraphNode> predicate) {
    Deque<CallGraphNode> queue = new ArrayDeque<>();
    Set<String> visited = new HashSet<>();

    queue.addAll(callGraph.getUsages(startFqn));
    while (!queue.isEmpty() && visited.size() <= maxVisited) {
      CallGraphNode current = queue.pop();
      if (!visited.contains(current.fqn())) {
        if (predicate.test(current)) {
          return TriBool.TRUE;
        }
        visited.add(current.fqn());
        queue.addAll(callGraph.getUsages(current.fqn()));
      }
    }

    if (queue.isEmpty()) {
      return TriBool.FALSE;
    } else {
      // We reached the maxVisited limit
      return TriBool.UNKNOWN; 
    }
  }

}
