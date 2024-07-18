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
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.visualization.GraphVisualizer;

public class ProjectLevelSymbolTableVisitor implements GraphVisualizer.TypeToGraphCollector<ProjectLevelSymbolTable> {
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
  public void parse(GraphVisualizer.Root<ProjectLevelSymbolTable> root) {
    nodes.add(new GraphVisualizer.Node(root.rootName(), root.rootName(), "color=\"red\", style=\"filled\", shape=\"box\""));
    parse(root.root(), root.rootName(), "");
  }

  private void parse(ProjectLevelSymbolTable root, String rootName, String parentLabel) {
    var globalDescriptorsByModuleName = root.globalDescriptorsByModuleName;
    for (Map.Entry<String, Set<Descriptor>> entry : globalDescriptorsByModuleName.entrySet()) {
      String moduleName = entry.getKey();
      Set<Descriptor> descriptors = entry.getValue();
      String currentNode = Integer.toString(System.identityHashCode(descriptors));
      nodes.add(new GraphVisualizer.Node(currentNode, "Module: " + moduleName, "color=\"yellow\", style=\"filled\", shape=\"box\""));
      edges.add(new GraphVisualizer.Edge(rootName, currentNode, parentLabel));
      for (Descriptor descriptor : descriptors) {
        parse(descriptor, currentNode, "descriptor");
      }
    }
  }

  private void parse(Descriptor descriptor, String currentNode, String parentLabel) {
    String descriptorNode = Integer.toString(System.identityHashCode(descriptor));
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

    edges.add(new GraphVisualizer.Edge(currentNode, descriptorNode, parentLabel));
  }

  private void parse(VariableDescriptor descriptor, String currentNode) {
    nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
      .addLabel("VariableDescriptor")
      .addLabel("name", descriptor.name())
      .addLabel("FQN", descriptor.fullyQualifiedName())
      .extraProp("style", "filled")
      .extraProp("fillcolor", "lightblue")
      .build());
  }

  private void parse(FunctionDescriptor descriptor, String currentNode) {
    // FQN, parameters, isAsynchronous, isInstanceMethod, decorators, hasDecorators, annotatedReturnTypeName
    nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
      .addLabel("FunctionDescriptor")
      .addLabel("name", descriptor.name())
      .addLabel("FQN", descriptor.fullyQualifiedName())
      .addLabel("parameters", Integer.toString(descriptor.parameters().size()))
      .addLabel("isAsynchronous", Boolean.toString(descriptor.isAsynchronous()))
      .addLabel("isInstanceMethod", Boolean.toString(descriptor.isInstanceMethod()))
      .addLabel("decorators", Integer.toString(descriptor.decorators().size()))
      .addLabel("hasDecorators", Boolean.toString(descriptor.hasDecorators()))
      .addLabel("annotatedReturnTypeName", Optional.ofNullable(descriptor.annotatedReturnTypeName()).orElse("Ø"))
      .extraProp("style", "filled")
      .extraProp("fillcolor", "orange")
      .build());

    for (FunctionDescriptor.Parameter parameter : descriptor.parameters()) {
      parse(parameter, currentNode);
    }
  }

  private void parse(FunctionDescriptor.Parameter parameter, String currentNode) {
    String parameterNode = Integer.toString(System.identityHashCode(parameter));
    // name, annotatedType, hasDefaultValue, isKeywordVariadic, isPositionalVariadic, isKeyworkOnly, isPositionalOnly
    nodes.add(new GraphVisualizer.NodeBuilder(parameterNode)
      .addLabel("Parameter")
      .addLabel("name", parameter.name())
      .addLabel("annotatedType", Optional.ofNullable(parameter.annotatedType()).orElse("Ø"))
      .addLabel("hasDefaultValue", Boolean.toString(parameter.hasDefaultValue()))
      .addLabel("isKeywordVariadic", Boolean.toString(parameter.isKeywordVariadic()))
      .addLabel("isPositionalVariadic", Boolean.toString(parameter.isPositionalVariadic()))
      .addLabel("isKeywordOnly", Boolean.toString(parameter.isKeywordOnly()))
      .addLabel("isPositionalOnly", Boolean.toString(parameter.isPositionalOnly()))
      .extraProp("style", "filled")
      .extraProp("fillcolor", "green")
      .build());
    edges.add(new GraphVisualizer.Edge(currentNode, parameterNode, "parameter"));
  }

  private void parse(ClassDescriptor descriptor, String currentNode) {
    // name, fullyQualifiedName, superClasses, members, hasDecorators, definitionLocation, hasSuperClassWithoutDescriptor, hasMetaClass,
    // metaclassFQN, supportsGenerics
    nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
      .addLabel("ClassDescriptor")
      .addLabel("name", descriptor.name())
      .addLabel("fullyQualifiedName", descriptor.fullyQualifiedName())
      .addLabel("superClasses", Integer.toString(descriptor.superClasses().size()))
      .addLabel("members", Integer.toString(descriptor.members().size()))
      .addLabel("hasDecorators", Boolean.toString(descriptor.hasDecorators()))
      .addLabel("definitionLocation", descriptor.definitionLocation() == null ? "null" : "not null")
      .addLabel("hasSuperClassWithoutDescriptor", Boolean.toString(descriptor.hasSuperClassWithoutDescriptor()))
      .addLabel("hasMetaClass", Boolean.toString(descriptor.hasMetaClass()))
      .addLabel("metaclassFQN", Optional.ofNullable(descriptor.metaclassFQN()).orElse("Ø"))
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
    nodes.add(new GraphVisualizer.NodeBuilder(currentNode)
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
