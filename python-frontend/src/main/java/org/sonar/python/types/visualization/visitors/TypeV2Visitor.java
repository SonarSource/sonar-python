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
package org.sonar.python.types.visualization.visitors;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;
import org.sonar.python.types.v2.UnknownType;
import org.sonar.python.types.visualization.GraphVisualizer;

import static org.sonar.python.types.visualization.GraphVisualizer.branchLimit;

public class TypeV2Visitor {
  public static class V2SymbolVisitor implements GraphVisualizer.TypeToGraphCollector<SymbolV2> {
    private final Set<GraphVisualizer.Node> nodes = new HashSet<>();
    private final Set<GraphVisualizer.Edge> edges = new HashSet<>();

    @Override
    public Set<GraphVisualizer.Edge> edges() {
      return edges;
    }

    @Override
    public Set<GraphVisualizer.Node> nodes() {
      return nodes;
    }

    @Override
    public void parse(GraphVisualizer.Root<SymbolV2> root) {
      nodes.add(new GraphVisualizer.Node(root.rootName(), root.rootName(), "color=\"yellow\", style=\"filled\", shape=\"box\""));
      parse(root.root(), root.rootName(), "");
    }

    private void parse(SymbolV2 symbol, String parentNode, String parentLabel) {
      String currentNode = Integer.toString(System.identityHashCode(symbol));
      nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
        .addLabel("SymbolV2")
        .addLabel("name", symbol.name())
        .addLabel("hasSingleBindingUsage", Boolean.toString(symbol.hasSingleBindingUsage()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "darkseagreen1")
        .build());
      edges.add(new GraphVisualizer.Edge(parentNode, currentNode, parentLabel));

      for (UsageV2 usage : symbol.usages()) {
        parse(usage, currentNode, "usage");
      }
    }

    private void parse(UsageV2 usage, String currentNode, String parentLabel) {
      String usageNode = Integer.toString(System.identityHashCode(usage));
      nodes.add(new GraphVisualizer.NodeBuilder(usageNode)
        .addLabel("UsageV2")
        .addLabel("kind", usage.kind().name())
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightcoral")
        .build());
      edges.add(new GraphVisualizer.Edge(currentNode, usageNode, parentLabel));
    }
  }

  public static class V2TypeInferenceVisitor implements GraphVisualizer.TypeToGraphCollector<PythonType> {
    private final Set<GraphVisualizer.Node> nodes = new HashSet<>();
    private final Set<GraphVisualizer.Edge> edges = new HashSet<>();

    private boolean filterDunder;
    private Integer branchingLimit;
    private Optional<Integer> depthLimit;
    private boolean skipBuiltins;

    public V2TypeInferenceVisitor(boolean filterDunder, @Nullable Integer branchingLimit, @Nullable Integer depthLimit, boolean skipBuiltins) {
      this.filterDunder = filterDunder;
      this.branchingLimit = branchingLimit;
      this.depthLimit = Optional.ofNullable(depthLimit);
      this.skipBuiltins = skipBuiltins;
    }

    public V2TypeInferenceVisitor() {
      this(true, 2, null, true);
    }

    @Override
    public Set<GraphVisualizer.Edge> edges() {
      return edges;
    }

    @Override
    public Set<GraphVisualizer.Node> nodes() {
      return nodes;
    }

    @Override
    public void parse(GraphVisualizer.Root<PythonType> root) {
      nodes.add(new GraphVisualizer.Node(root.rootName(), root.rootName(), "color=\"yellow\", style=\"filled\", shape=\"box\""));
      parse(root.root(), root.rootName(), "", 0);
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
        parse(unknownType, currentNode, depth + 1);
      } else if (type instanceof UnionType unionType) {
        parse(unionType, currentNode, depth + 1);
      } else {
        throw new IllegalStateException("Unsupported type: " + type.getClass().getName());
      }
      edges.add(new GraphVisualizer.Edge(parentNode, currentNode, parentLabel));
    }

    private void parse(ParameterV2 type, String parentNode, String parentLabel, int depth) {
      String currentNode = Integer.toString(System.identityHashCode(type));
      // name, hasDefault, isKeywordOnly, isPositionalOnly, isKeywordVariadic, isPositionalVariadic
      edges.add(new GraphVisualizer.Edge(parentNode, currentNode, parentLabel));

      this.nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
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
      this.nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
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
      this.nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
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
      for (PythonType superClass : branchLimit(type.superClasses().stream(), branchingLimit).toList()) {
        parse(superClass, currentNode, "superClass", depth + 1);
      }

      for (PythonType metaClass : branchLimit(type.metaClasses().stream(), branchingLimit).toList()) {
        parse(metaClass, currentNode, "metaClass", depth + 1);
      }

      for (PythonType attribute : branchLimit(type.attributes().stream(), branchingLimit).toList()) {
        parse(attribute, currentNode, "attribute", depth + 1);
      }

      for (Member member : branchLimit(type.members().stream().filter(member -> !filterDunder || !member.name().startsWith("__")), branchingLimit).toList()) {
        parse(member.type(), currentNode, "member", depth + 1);
      }

    }

    private void parse(FunctionType type, String currentNode, int depth) {
      this.nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
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
      this.nodes.add(new GraphVisualizer.Node(
        currentNode,
        "UnknownType"));
    }

    private void parse(UnionType type, String currentNode, int depth) {
      this.nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
        .addLabel("UnionType")
        .addLabel("candidates", Integer.toString(type.candidates().size()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "orangered2")
        .build());

      for (PythonType candidate : type.candidates()) {
        parse(candidate, currentNode, "candidate", depth + 1);
      }
    }

  }

}
