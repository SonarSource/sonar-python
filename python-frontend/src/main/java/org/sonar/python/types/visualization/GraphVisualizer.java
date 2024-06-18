/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.types.visualization;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

public class GraphVisualizer {

  public interface TypeToGraphCollector<R> {
    Set<Edge> edges();

    Set<Node> nodes();

    void parse(Root<R> root);
  }

  public record Root<T>(T root, String rootName) {
  }


  public static class NodeBuilder {
    private final String name;
    private final Collection<Label> labels = new ArrayList<>();
    private final Map<String, String> extraProps = new HashMap<>();

    record Label(String key, @Nullable String value) {
      @Override
      public String toString() {
        if (value == null) {
          return key;
        }
        return "{" + key + " | " + value + "}";
      }
    }

    public NodeBuilder(String name) {
      this.name = name;
    }

    public NodeBuilder addLabel(String key, String value) {
      labels.add(new Label(key, value));
      return this;
    }

    public NodeBuilder addLabel(String key) {
      labels.add(new Label(key, null));
      return this;
    }

    public NodeBuilder color(String color) {
      extraProps.put("color", color);
      return this;
    }

    public NodeBuilder extraProp(String key, String value) {
      extraProps.put(key, value);
      return this;
    }

    public Node build() {
      if (extraProps.isEmpty()) {
        return new Node(name, labels.stream().sorted().map(Label::toString).collect(Collectors.joining(" | ")));
      }
      return new Node(name, labels.stream().map(Label::toString).collect(Collectors.joining(" | ")),
        extraProps.entrySet().stream().map(e -> e.getKey() + "=\"" + e.getValue() + "\"").collect(Collectors.joining(", ")));
    }
  }

  final Set<Node> nodes;

  final Set<Edge> edges;

  public static class Builder {

    private final Map<TypeToGraphCollector, Root> collectors = new HashMap<>();

    public <R> Builder addCollector(TypeToGraphCollector<R> collector, Root<R> root) {
      collectors.put(collector, root);
      return this;
    }

    public GraphVisualizer build() {
      GraphVisualizer graph = new GraphVisualizer();
      for (Map.Entry<TypeToGraphCollector, Root> entry : collectors.entrySet()) {
        entry.getKey().parse(entry.getValue());
        graph.nodes.addAll(entry.getKey().nodes());
        graph.edges.addAll(entry.getKey().edges());
      }
      return graph;
    }

  }

  public record Node(String name, String label, String extraProps) {
    @Override
    public String toString() {
      String formattedExtraProps = extraProps.isEmpty() ? "" : (", " + extraProps);
      return name + " [label=\"" + label + "\"" + formattedExtraProps + "]";
    }

    public Node(String name, String label) {
      this(name, label, "");
    }
  }

  public record Edge(String from, String to, String label) {
    @Override
    public String toString() {
      if (!label.isEmpty()) {
        return from + " -> " + to + " [label=\"" + label + "\"]";
      }
      return from + " -> " + to;
    }

    Edge(String from, String to) {
      this(from, to, "");
    }
  }

  public GraphVisualizer() {
    this.nodes = new LinkedHashSet<>();
    this.edges = new LinkedHashSet<>();
  }

  @Override
  public String toString() {
    StringBuilder output = new StringBuilder("digraph G {\n");
    output.append("""
      node [
        shape = record
      ]
      graph [
        rankdir = "LR"
      ]
      """);

    for (Node node : nodes) {
      output.append(node).append("\n");
    }
    for (Edge edge : edges) {
      output.append(edge).append("\n");
    }
    output.append("}");

    return output.toString();
  }
}

