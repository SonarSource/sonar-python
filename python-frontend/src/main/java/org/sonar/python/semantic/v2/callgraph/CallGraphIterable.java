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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.function.BiFunction;

public class CallGraphIterable implements Iterable<CallGraphNode> {
  private final String startFqn;
  private final CallGraph callGraph;
  private final BiFunction<CallGraph, String, List<CallGraphNode>> getUsages;

  public CallGraphIterable(String startFqn, CallGraph callGraph, BiFunction<CallGraph, String, List<CallGraphNode>> getUsages) {
    this.startFqn = startFqn;
    this.callGraph = callGraph;
    this.getUsages = getUsages;
  }

  @Override
  public Iterator<CallGraphNode> iterator() {
    return new CallGraphIterator();
  }

  private class CallGraphIterator implements Iterator<CallGraphNode> {
    private final Set<String> visited = new HashSet<>();
    private final Deque<CallGraphNode> queue = new ArrayDeque<>();

    public CallGraphIterator() {
      queue.addAll(getUsages.apply(callGraph, startFqn));
    }

    @Override
    public boolean hasNext() {
      return queue.stream().anyMatch(node -> !visited.contains(node.fqn()));
    }

    @Override
    public CallGraphNode next() {
      if (queue.isEmpty()) {
        throw new NoSuchElementException("No more elements to iterate");
      }

      CallGraphNode current = null;
      do {
        current = queue.pop();
      } while (!queue.isEmpty() && visited.contains(current.fqn()));

      if (visited.contains(current.fqn())) {
        throw new NoSuchElementException("No more elements to iterate");
      }

      queue.addAll(getUsages.apply(callGraph, current.fqn()));

      visited.add(current.fqn());
      return current;
    }
  }
}
