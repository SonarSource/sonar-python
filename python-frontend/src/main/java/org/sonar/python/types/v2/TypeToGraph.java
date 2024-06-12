package org.sonar.python.types.v2;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.Nullable;

public class TypeToGraph {

  public interface TypeToGraphCollector<R> {
    Set<Edge> edges();

    Set<Node> nodes();

    void parse(Root<R> root);
  }

  public record Root<T>(T root, String rootName) {
  }

  public static class V2TypeInferenceVisitor implements TypeToGraphCollector<PythonType> {
    private final Set<Node> nodes = new HashSet<>();
    private final Set<Edge> edges = new HashSet<>();

    private boolean filterDunder = true;
    private Optional<Integer> branchingLimit = Optional.of(2);
    private Optional<Integer> depthLimit = Optional.empty();

    public V2TypeInferenceVisitor(boolean filterDunder, @Nullable Integer branchingLimit, @Nullable Integer depthLimit) {
      this.filterDunder = filterDunder;
      this.branchingLimit = Optional.ofNullable(branchingLimit);
      this.depthLimit = Optional.ofNullable(depthLimit);
    }

    public V2TypeInferenceVisitor() {
      this(true, 2, null);
    }

    @Override
    public Set<Edge> edges() {
      return edges;
    }

    @Override
    public Set<Node> nodes() {
      return nodes;
    }

    @Override
    public void parse(Root<PythonType> root) {
      nodes.add(new Node(root.rootName, root.rootName, "color=\"red\", style=\"filled\", shape=\"box\""));
      parse(root.root, root.rootName, "", 0);
    }

    private void parse(PythonType type, String parentNode, String parentLabel, int depth) {
      String currentNode = Integer.toString(System.identityHashCode(type));

      if (depthLimit.isPresent() && depth >= depthLimit.get()) {
        return;
      }

      if (type instanceof ObjectType objectType) {
        parse(objectType, currentNode, depth + 1);
      } else if (type instanceof ClassType functionType) {
        parse(functionType, currentNode, depth + 1);
      } else if (type instanceof FunctionType functionType) {
        parse(functionType, currentNode, depth + 1);
      } else if (type instanceof UnknownType unknownType) {
        // parse(unknownType, parentNode, parentLabel, currentNode, depth + 1);
        return;
      } else if (type instanceof UnionType unionType) {
        parse(unionType, currentNode, depth + 1);
      } else {
        throw new IllegalStateException("Unsupported type: " + type.getClass().getName());
      }
      edges.add(new Edge(parentNode, currentNode, parentLabel));
    }

    private void parse(ParameterV2 type, String parentNode, String parentLabel, int depth) {
      String currentNode = Integer.toString(System.identityHashCode(type));
      // name, hasDefault, isKeywordOnly, isPositionalOnly, isKeywordVariadic, isPositionalVariadic
      edges.add(new Edge(parentNode, currentNode, parentLabel));
      this.nodes.add(new Node(
        currentNode,
        "ParameterV2 | {name | " + type.name() + "}" + " | {hasDefault | " + type.hasDefaultValue() + "}" + " | {isKeywordOnly | " + type.isKeywordOnly() + "}"
          + " | {isPositionalOnly | "
          + type.isPositionalOnly() + "}" + " | {isKeywordVariadic | " + type.isKeywordVariadic() + "}" + " | {isPositionalVariadic | " + type.isPositionalVariadic() + "}"));

      parse(type.declaredType(), currentNode, "declaredType", depth + 1);
    }

    private void parse(ObjectType type, String currentNode, int depth) {
      this.nodes.add(new Node(
        currentNode,
        "ObjectType | {attributes | " + type.attributes().size() + "}"));

      for (PythonType attribute : type.attributes()) {
        parse(attribute, currentNode, "attribute", depth + 1);
      }
      parse(type.type(), currentNode, "type", depth + 1);
    }

    private <T> Stream<T> branchLimit(Stream<T> stream) {
      if (branchingLimit.isPresent()) {
        return stream.limit(branchingLimit.get());
      }
      return stream;
    }

    private void parse(ClassType type, String currentNode, int depth) {
      // name, members, superClass, metaClass, hasDecorators, attributes
      this.nodes.add(new Node(
        currentNode,
        "ClassType | {name | " + type.name() + "} | {members | " + type.members().size() + "}" + " | {superClasses | " + type.superClasses().size() + "}" + " | {metaClasses | "
          + type.metaClasses().size() + "}" + " | {attributes | " + type.attributes().size() + "}"));

      for (PythonType superClass : branchLimit(type.superClasses().stream()).toList()) {
        parse(superClass, currentNode, "superClass", depth + 1);
      }

      for (PythonType metaClass : branchLimit(type.metaClasses().stream()).toList()) {
        parse(metaClass, currentNode, "metaClass", depth + 1);
      }

      for (Member member : branchLimit(type.members().stream().filter(member -> !filterDunder || !member.name().startsWith("__"))).toList()) {
        parse(member.type(), currentNode, "member", depth + 1);
      }

      for (PythonType attribute : branchLimit(type.attributes().stream()).toList()) {
        parse(attribute, currentNode, "attribute", depth + 1);
      }
    }

    private void parse(FunctionType type, String currentNode, int depth) {
      // name, parameters, isAsynchronous, hasDecorators, isInstanceMethod, hasVariadicParameter

      this.nodes.add(new Node(
        currentNode,
        "FunctionType | {name | " + type.name() + "} | {parameters | " + type.parameters().size() + "}" + " | {isAsynchronous | " + type.isAsynchronous() + "}"
          + " | {hasDecorators | "
          + type.hasDecorators() + "}" + " | {isInstanceMethod | " + type.isInstanceMethod() + "}" + " | {hasVariadicParameter | " + type.hasVariadicParameter() + "}"));

      for (PythonType attribute : type.attributes()) {
        parse(attribute, currentNode, "attribute", depth + 1);
      }

      for (ParameterV2 parameter : type.parameters()) {
        parse(parameter, currentNode, "parameter", depth + 1);
      }

      parse(type.returnType(), currentNode, "returnType", depth + 1);
    }

    private void parse(UnknownType type, String currentNode, int depth) {
      this.nodes.add(new Node(
        currentNode,
        "UnknownType"));
    }

    private void parse(UnionType type, String currentNode, int depth) {
      this.nodes.add(new Node(
        currentNode,
        "UnionType | {candidates | " + type.candidates().size() + "}"));

      for (PythonType candidate : type.candidates()) {
        parse(candidate, currentNode, "candidate", depth + 1);
      }
    }

  }

  private final Set<Node> nodes;

  private final Set<Edge> edges;

  public static class Builder {

    private final Map<TypeToGraphCollector, Root> collectors = new HashMap<>();

    public <R> Builder addCollector(TypeToGraphCollector<R> collector, Root<R> root) {
      collectors.put(collector, root);
      return this;
    }

    public TypeToGraph build() {
      TypeToGraph graph = new TypeToGraph();
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

    Node(String name, String label) {
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

  public TypeToGraph() {
    this.nodes = new HashSet<>();
    this.edges = new HashSet<>();
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
