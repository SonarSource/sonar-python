package org.sonar.python.types.v2;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class TypeToGraph {

  private final Set<Node> nodes;

  private final Set<Edge> edges;

  private final boolean filterDunder = true;

  public static class Builder {

    private final List<Root> roots = new ArrayList<>();

    public Builder() {
    }

    public Builder root(PythonType root, String rootName) {
      roots.add(new Root(root, rootName));
      return this;
    }

    public TypeToGraph build() {
      TypeToGraph graph = new TypeToGraph();
      for (Root root : roots) {
        graph.parse(root);
      }
      return graph;
    }
  }

  record Node(String name, String label) {
    @Override
    public String toString() {
      return name + " [label=\"" + label + "\"]";
    }
  }

  record Root(PythonType root, String rootName) {
  }

  record Edge(String from, String to, String label) {
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

  public TypeToGraph() {
    this.nodes = new HashSet<>();
    this.edges = new HashSet<>();
  }

  public void parse(Root root) {
    nodes.add(new Node(root.rootName, root.rootName));
    this.parse(root.root, root.rootName, "");
  }

  private void parse(PythonType type, String parentNode, String parentLabel) {
    String currentNode = Integer.toString(System.identityHashCode(type));

    if (type instanceof ObjectType objectType) {
      parse(objectType, parentNode, parentLabel, currentNode);
    } else if (type instanceof ClassType functionType) {
      parse(functionType, parentNode, parentLabel, currentNode);
    } else if (type instanceof FunctionType functionType) {
      parse(functionType, parentNode, parentLabel, currentNode);
    } else if (type instanceof UnknownType unknownType) {
//      parse(unknownType, parentNode, parentLabel, currentNode);
      return;
    } else if (type instanceof UnionType unionType) {
      parse(unionType, parentNode, parentLabel, currentNode);
    } else {
      throw new IllegalStateException("Unsupported type: " + type.getClass().getName());
    }
    edges.add(new Edge(parentNode, currentNode, parentLabel));
  }

  private void parse(ParameterV2 type, String parentNode, String parentLabel) {
    String currentNode = Integer.toString(System.identityHashCode(type));

    edges.add(new Edge(parentNode, currentNode, parentLabel));
    this.nodes.add(new Node(
      currentNode,
      "ParameterV2 | {name | " + type.name() + "}"));

    parse(type.declaredType(), currentNode, "declaredType");
  }

  private void parse(ObjectType type, String parentNode, String parentLabel, String currentNode) {
    this.nodes.add(new Node(
      currentNode,
      "ObjectType | {attributes | " + type.attributes().size() + "}"));

    for (PythonType attribute : type.attributes()) {
      parse(attribute, currentNode, "attribute");
    }
    parse(type.type(), currentNode, "type");
  }

  private void parse(ClassType type, String parentNode, String parentLabel, String currentNode) {
    this.nodes.add(new Node(
      currentNode,
      "ClassType | {name | " + type.name() + "} | {members | " + type.members().size() + "}"));

    for (PythonType superClass : type.superClasses()) {
      parse(superClass, currentNode, "superClass");
    }

    for (PythonType metaClass : type.metaClasses()) {
      parse(metaClass, currentNode, "metaClass");
    }

    for (Member member : type.members().stream().filter(member -> !filterDunder || !member.name().startsWith("__")).toList()) {
      parse(member.type(), currentNode, "member");
    }
  }

  private void parse(FunctionType type, String parentNode, String parentLabel, String currentNode) {
    this.nodes.add(new Node(
      currentNode,
      "FunctionType | {name | " + type.name() + "} | {parameters | " + type.parameters().size() + "}"));

    for (PythonType attribute : type.attributes()) {
      parse(attribute, currentNode, "attribute");
    }

    for (ParameterV2 parameter : type.parameters()) {
      parse(parameter, currentNode, "parameter");
    }

    parse(type.returnType(), currentNode, "returnType");
  }

  private void parse(UnknownType type, String parentNode, String parentLabel, String currentNode) {
    this.nodes.add(new Node(
      currentNode,
      "UnknownType"));
  }

  private void parse(UnionType type, String parentNode, String parentLabel, String currentNode) {
    this.nodes.add(new Node(
      currentNode,
      "UnionType | {candidates | " + type.candidates().size() + "}"));

    for (PythonType candidate : type.candidates()) {
      parse(candidate, currentNode, "candidate");
    }
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
