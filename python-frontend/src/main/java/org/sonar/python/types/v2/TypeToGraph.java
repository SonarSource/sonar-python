package org.sonar.python.types.v2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;

public class TypeToGraph {

  public interface TypeToGraphCollector<R> {
    Set<Edge> edges();

    Set<Node> nodes();

    void parse(Root<R> root);
  }

  public record Root<T>(T root, String rootName) {
  }

  public static class ProjectLevelSymbolTableVisitor implements TypeToGraphCollector<ProjectLevelSymbolTable> {
    private final Set<Node> nodes = new HashSet<>();
    private final Set<Edge> edges = new HashSet<>();

    @Override
    public Set<Edge> edges() {
      return edges;
    }

    @Override
    public Set<Node> nodes() {
      return nodes;
    }
    @Override
    public void parse(Root<ProjectLevelSymbolTable> root) {
      nodes.add(new Node(root.rootName, root.rootName, "color=\"yellow\", style=\"filled\", shape=\"box\""));
      parse(root.root, root.rootName, "");
    }

    private void parse(ProjectLevelSymbolTable root, String rootName, String parentLabel) {
      var globalDescriptorsByModuleName = root.globalDescriptorsByModuleName;
      for (Map.Entry<String, Set<Descriptor>> entry : globalDescriptorsByModuleName.entrySet()) {
        String moduleName = entry.getKey();
        Set<Descriptor> descriptors = entry.getValue();
        String currentNode = Integer.toString(System.identityHashCode(descriptors));
        nodes.add(new Node(currentNode, "Module: " + moduleName, "color=\"yellow\", style=\"filled\", shape=\"box\""));
        edges.add(new Edge(rootName, currentNode, parentLabel));
        for (Descriptor descriptor : descriptors) {
          parse(descriptor, currentNode, "descriptor");
        }
      }
    }

    private void parse(Descriptor descriptor, String currentNode, String parentLabel) {
      String descriptorNode = Integer.toString(System.identityHashCode(descriptor));
      nodes.add(new Node(descriptorNode, "Descriptor", "color=\"yellow\", style=\"filled\", shape=\"box\""));
      if (descriptor instanceof VariableDescriptor variableDescriptor) {
        parse(variableDescriptor, descriptorNode);
      } else if (descriptor instanceof FunctionDescriptor functionDescriptor) {
        parse(functionDescriptor, descriptorNode);
      } else if (descriptor instanceof AmbiguousDescriptor ambiguousDescriptor) {
        parse(ambiguousDescriptor, descriptorNode);
      } else if (descriptor instanceof ClassDescriptor classDescriptor) {
        parse(classDescriptor, descriptorNode);
      } else {
        throw new IllegalStateException("Unsupported descriptor: " + descriptor.getClass().getName());
      }

      edges.add(new Edge(currentNode, descriptorNode, parentLabel));
    }

    private void parse(VariableDescriptor descriptor, String currentNode) {
      nodes.add(new NodeBuilder(currentNode)
        .addLabel("VariableDescriptor")
        .addLabel("name", descriptor.name())
        .addLabel("FQN", descriptor.fullyQualifiedName())
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightblue")
        .build());
    }

    private void parse(FunctionDescriptor descriptor, String currentNode) {
      // FQN, parameters, isAsynchronous, isInstanceMethod, decorators, hasDecorators, annotatedReturnTypeName
      nodes.add(new NodeBuilder(currentNode)
        .addLabel("FunctionDescriptor")
        .addLabel("name", descriptor.name())
        .addLabel("FQN", descriptor.fullyQualifiedName())
        .addLabel("parameters", Integer.toString(descriptor.parameters().size()))
        .addLabel("isAsynchronous", Boolean.toString(descriptor.isAsynchronous()))
        .addLabel("isInstanceMethod", Boolean.toString(descriptor.isInstanceMethod()))
        .addLabel("decorators", Integer.toString(descriptor.decorators().size()))
        .addLabel("hasDecorators", Boolean.toString(descriptor.hasDecorators()))
        .addLabel("annotatedReturnTypeName", descriptor.annotatedReturnTypeName())
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightorange")
        .build());

      for (FunctionDescriptor.Parameter parameter : descriptor.parameters()) {
        parse(parameter, currentNode);
      }
    }

    private void parse(FunctionDescriptor.Parameter parameter, String currentNode) {
      String parameterNode = Integer.toString(System.identityHashCode(parameter));
      // name, annotatedType, hasDefaultValue, isKeywordVariadic, isPositionalVariadic, isKeyworkOnly, isPositionalOnly
      nodes.add(new NodeBuilder(parameterNode)
        .addLabel("Parameter")
        .addLabel("name", parameter.name())
        .addLabel("annotatedType", parameter.annotatedType())
        .addLabel("hasDefaultValue", Boolean.toString(parameter.hasDefaultValue()))
        .addLabel("isKeywordVariadic", Boolean.toString(parameter.isKeywordVariadic()))
        .addLabel("isPositionalVariadic", Boolean.toString(parameter.isPositionalVariadic()))
        .addLabel("isKeywordOnly", Boolean.toString(parameter.isKeywordOnly()))
        .addLabel("isPositionalOnly", Boolean.toString(parameter.isPositionalOnly()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightgreen")
        .build());
      edges.add(new Edge(currentNode, parameterNode, "parameter"));
    }

    private void parse(ClassDescriptor descriptor, String currentNode) {
      // name, fullyQualifiedName, superClasses, members, hasDecorators, definitionLocation, hasSuperClassWithoutDescriptor, hasMetaClass,
      // metaclassFQN, supportsGenerics
      nodes.add(new NodeBuilder(currentNode)
        .addLabel("ClassDescriptor")
        .addLabel("name", descriptor.name())
        .addLabel("fullyQualifiedName", descriptor.fullyQualifiedName())
        .addLabel("superClasses", Integer.toString(descriptor.superClasses().size()))
        .addLabel("members", Integer.toString(descriptor.members().size()))
        .addLabel("hasDecorators", Boolean.toString(descriptor.hasDecorators()))
        .addLabel("definitionLocation", descriptor.definitionLocation().toString())
        .addLabel("hasSuperClassWithoutDescriptor", Boolean.toString(descriptor.hasSuperClassWithoutDescriptor()))
        .addLabel("hasMetaClass", Boolean.toString(descriptor.hasMetaClass()))
        .addLabel("metaclassFQN", descriptor.metaclassFQN())
        .addLabel("supportsGenerics", Boolean.toString(descriptor.supportsGenerics()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightpink")
        .build());

      for (Descriptor member : descriptor.members()) {
        parse(member, currentNode, "member");
      }
    }

    private void parse(AmbiguousDescriptor descriptor, String currentNode) {
      // descriptors
      nodes.add(new NodeBuilder(currentNode)
        .addLabel("AmbiguousDescriptor")
        .addLabel("descriptors", Integer.toString(descriptor.alternatives().size()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightcoral")
        .build());

      for (Descriptor desc : descriptor.alternatives()) {
        parse(desc, currentNode, "descriptor");
      }
    }
  }

  public static class V2SymbolVisitor implements TypeToGraphCollector<SymbolV2> {
    private final Set<Node> nodes = new HashSet<>();
    private final Set<Edge> edges = new HashSet<>();

    @Override
    public Set<Edge> edges() {
      return edges;
    }

    @Override
    public Set<Node> nodes() {
      return nodes;
    }

    @Override
    public void parse(Root<SymbolV2> root) {
      nodes.add(new Node(root.rootName, root.rootName, "color=\"yellow\", style=\"filled\", shape=\"box\""));
      parse(root.root, root.rootName, "");
    }

    private void parse(SymbolV2 symbol, String parentNode, String parentLabel) {
      String currentNode = Integer.toString(System.identityHashCode(symbol));
      nodes.add(new NodeBuilder(currentNode)
        .addLabel("SymbolV2")
        .addLabel("name", symbol.name())
        .addLabel("hasSingleBindingUsage", Boolean.toString(symbol.hasSingleBindingUsage()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "darkseagreen1")
        .build());
      edges.add(new Edge(parentNode, currentNode, parentLabel));

      for (UsageV2 usage : symbol.usages()) {
        parse(usage, currentNode, "usage");
      }
    }

    private void parse(UsageV2 usage, String currentNode, String parentLabel) {
      String usageNode = Integer.toString(System.identityHashCode(usage));
      nodes.add(new NodeBuilder(usageNode)
        .addLabel("UsageV2")
        .addLabel("kind", usage.kind().name())
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightcoral")
        .build());
      edges.add(new Edge(currentNode, usageNode, parentLabel));
    }
  }

  public static class V2TypeInferenceVisitor implements TypeToGraphCollector<PythonType> {
    private final Set<Node> nodes = new HashSet<>();
    private final Set<Edge> edges = new HashSet<>();

    private boolean filterDunder;
    private Optional<Integer> branchingLimit;
    private Optional<Integer> depthLimit;
    private boolean skipBuiltins;

    public V2TypeInferenceVisitor(boolean filterDunder, @Nullable Integer branchingLimit, @Nullable Integer depthLimit, boolean skipBuiltins) {
      this.filterDunder = filterDunder;
      this.branchingLimit = Optional.ofNullable(branchingLimit);
      this.depthLimit = Optional.ofNullable(depthLimit);
      this.skipBuiltins = skipBuiltins;
    }

    public V2TypeInferenceVisitor() {
      this(true, 2, null, true);
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
      nodes.add(new Node(root.rootName, root.rootName, "color=\"yellow\", style=\"filled\", shape=\"box\""));
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

      this.nodes.add(new NodeBuilder(currentNode)
        .addLabel("ParameterV2")
        .addLabel("name", type.name() == null ? "null" : type.name())
        .addLabel("hasDefault", Boolean.toString(type.hasDefaultValue()))
        .addLabel("isKeywordOnly", Boolean.toString(type.isKeywordOnly()))
        .addLabel("isPositionalOnly", Boolean.toString(type.isPositionalOnly()))
        .addLabel("isKeywordVariadic", Boolean.toString(type.isKeywordVariadic()))
        .addLabel("isPositionalVariadic", Boolean.toString(type.isPositionalVariadic()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "olivedrab1")
        .build());

      parse(type.declaredType(), currentNode, "declaredType", depth + 1);
    }

    private void parse(ObjectType type, String currentNode, int depth) {
      this.nodes.add(new NodeBuilder(currentNode)
        .addLabel("ObjectType")
        .addLabel("type", type.type().name())
        .addLabel("attributes", Integer.toString(type.attributes().size()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightsalmon")
        .build());

      for (PythonType attribute : type.attributes()) {
        parse(attribute, currentNode, "attribute", depth + 1);
      }
      parse(type.type(), currentNode, "type", depth + 1);
    }

    private void parse(ClassType type, String currentNode, int depth) {
      // name, members, superClass, metaClass, hasDecorators, attributes
      this.nodes.add(new NodeBuilder(currentNode)
        .addLabel("ClassType")
        .addLabel("name", type.name())
        .addLabel("members", Integer.toString(type.members().size()))
        .addLabel("superClasses", Integer.toString(type.superClasses().size()))
        .addLabel("metaClasses", Integer.toString(type.metaClasses().size()))
        .addLabel("hasDecorators", Boolean.toString(type.hasDecorators()))
        .addLabel("attributes", Integer.toString(type.attributes().size()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightblue")
        .build());

      if (skipBuiltins && type.definitionLocation().isEmpty()) {
        return;
      }
      for (PythonType superClass : branchLimit(type.superClasses().stream()).toList()) {
        parse(superClass, currentNode, "superClass", depth + 1);
      }

      for (PythonType metaClass : branchLimit(type.metaClasses().stream()).toList()) {
        parse(metaClass, currentNode, "metaClass", depth + 1);
      }

      for (PythonType attribute : branchLimit(type.attributes().stream()).toList()) {
        parse(attribute, currentNode, "attribute", depth + 1);
      }

      for (Member member : branchLimit(type.members().stream().filter(member -> !filterDunder || !member.name().startsWith("__"))).toList()) {
        parse(member.type(), currentNode, "member", depth + 1);
      }

    }

    private void parse(FunctionType type, String currentNode, int depth) {
      this.nodes.add(new NodeBuilder(currentNode)
        .addLabel("FunctionType")
        .addLabel("name", type.name())
        .addLabel("parameters", Integer.toString(type.parameters().size()))
        .addLabel("isAsynchronous", Boolean.toString(type.isAsynchronous()))
        .addLabel("hasDecorators", Boolean.toString(type.hasDecorators()))
        .addLabel("isInstanceMethod", Boolean.toString(type.isInstanceMethod()))
        .addLabel("hasVariadicParameter", Boolean.toString(type.hasVariadicParameter()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "mediumslateblue")
        .build());

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
      this.nodes.add(new TypeToGraph.NodeBuilder(currentNode)
        .addLabel("UnionType")
        .addLabel("candidates", Integer.toString(type.candidates().size()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "orangered2")
        .build());

      for (PythonType candidate : type.candidates()) {
        parse(candidate, currentNode, "candidate", depth + 1);
      }
    }

    private <T> Stream<T> branchLimit(Stream<T> stream) {
      if (branchingLimit.isPresent()) {
        return stream.limit(branchingLimit.get());
      }
      return stream;
    }

  }

  public static class NodeBuilder {
    private final String name;
    private final Collection<Label> labels = new ArrayList<>();
    private final Map<String, String> extraProps = new HashMap<>();

    record Label(String key, @Nullable String value) {
      @Override
      public String toString() {
        if (value == null) {
          return "{" + key + "}";
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
        return new Node(name, labels.stream().map(Label::toString).collect(Collectors.joining(" | ")));
      }
      return new Node(name, labels.stream().map(Label::toString).collect(Collectors.joining(" | ")),
        extraProps.entrySet().stream().map(e -> e.getKey() + "=\"" + e.getValue() + "\"").collect(Collectors.joining(", ")));
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
