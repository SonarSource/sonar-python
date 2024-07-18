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
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.AnyType;
import org.sonar.python.types.DeclaredType;
import org.sonar.python.types.RuntimeType;
import org.sonar.python.types.UnionType;
import org.sonar.python.types.UnknownClassType;
import org.sonar.python.types.visualization.GraphVisualizer;

public class TypeV1Visitor implements GraphVisualizer.TypeToGraphCollector<InferredType> {
  private final Set<GraphVisualizer.Node> nodes = new HashSet<>();
  private final Set<GraphVisualizer.Edge> edges = new HashSet<>();
  private boolean skipBuiltins = true;

  public TypeV1Visitor(boolean skipBuiltins) {
    this.skipBuiltins = skipBuiltins;
  }

  public TypeV1Visitor() {
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
  public void parse(GraphVisualizer.Root<InferredType> root) {
    nodes.add(new GraphVisualizer.Node(root.rootName(), root.rootName(), "color=\"red\", style=\"filled\", shape=\"box\""));
    parseType(root.root(), root.rootName(), "");
  }

  private void parseType(InferredType type, String currentNode, String parentLabel) {
    String typeNode = Integer.toString(System.identityHashCode(type));

    if (nodes.stream().anyMatch(n -> n.name().equals(typeNode))) {
      return;
    }

    if (type instanceof RuntimeType runtimeType) {
      parse(runtimeType, typeNode);
    } else if (type instanceof AnyType) {
      parse(typeNode);
    } else if (type instanceof DeclaredType declaredType) {
      parse(declaredType, typeNode);
    } else if (type instanceof UnionType unionType) {
      parse(unionType, typeNode);
    } else if (type instanceof UnknownClassType unknownClassType) {
      parse(unknownClassType, typeNode);
    } else {
      throw new IllegalStateException("Unsupported type: " + type.getClass().getName());
    }
    edges.add(new GraphVisualizer.Edge(currentNode, typeNode, parentLabel));
  }

  private void parse(UnknownClassType unknownClassType, String typeNode) {
    nodes.add(new GraphVisualizer.NodeBuilder(typeNode)
      .addLabel("UnknownClassType")
      .extraProp("style", "filled")
      .extraProp("fillcolor", "lightblue")
      .build());

    parseSymbol(unknownClassType.typeSymbol(), typeNode, "typeSymbol");
  }

  private void parse(UnionType unionType, String typeNode) {
    nodes.add(new GraphVisualizer.NodeBuilder(typeNode)
      .addLabel("UnionType")
      .addLabel("types", Integer.toString(unionType.types().size()))
      .extraProp("style", "filled")
      .extraProp("fillcolor", "orangered2")
      .build());
    unionType.types().forEach(type -> parseType(type, typeNode, "type"));
  }

  private void parse(DeclaredType declaredType, String typeNode) {

    try {
      var typeArgsField = DeclaredType.class.getDeclaredField("typeArgs");
      var builtinFullyQualifiedNameField = DeclaredType.class.getDeclaredField("builtinFullyQualifiedName");
      typeArgsField.setAccessible(true);
      builtinFullyQualifiedNameField.setAccessible(true);

      List<DeclaredType> typeArgs = (List<DeclaredType>) typeArgsField.get(declaredType);

      nodes.add(new GraphVisualizer.NodeBuilder(typeNode)
        .addLabel("DeclaredType")
        .addLabel("typeName", declaredType.typeName())
        .addLabel("typeClass", declaredType.getTypeClass().name())
        .addLabel("typeArgs", String.valueOf(typeArgs.size()))
        .addLabel("alternativeTypeSymbols", Integer.toString(declaredType.alternativeTypeSymbols().size()))
        .addLabel("builtinFullyQualifiedName", Optional.ofNullable(builtinFullyQualifiedNameField.get(declaredType)).orElse("Ø").toString())
        .addLabel("alternativeTypeSymbols", Integer.toString(declaredType.alternativeTypeSymbols().size()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightblue")
        .build());

      parseSymbol(declaredType.getTypeClass(), typeNode, "typeClass");
      typeArgs.forEach(typeArg -> parseType(typeArg, typeNode, "typeArg"));
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private void parseSymbol(Symbol symbol, String currentNode, String parentLabel) {
    String symbolNode = Integer.toString(System.identityHashCode(symbol));

    if (symbol instanceof ClassSymbol classSymbol) {
      nodes.add(new GraphVisualizer.NodeBuilder(symbolNode)
        .addLabel("ClassSymbol")
        .addLabel("name", classSymbol.name())
        .addLabel("fullyQualifiedName", Optional.ofNullable(classSymbol.fullyQualifiedName()).orElse("Ø"))
        .addLabel("superClasses", Integer.toString(classSymbol.superClasses().size()))
        .addLabel("members", Integer.toString(classSymbol.declaredMembers().size()))
        .addLabel("hasDecorators", Boolean.toString(classSymbol.hasDecorators()))
        .addLabel("hasMetaClass", Boolean.toString(classSymbol.hasMetaClass()))
        .addLabel("definitionLocation", classSymbol.definitionLocation() == null ? "null" : "not null")
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightblue")
        .build());
      if (!skipBuiltins || classSymbol.definitionLocation() != null) {
        classSymbol.superClasses().forEach(superClass -> parseSymbol(superClass, symbolNode, "superClass"));
        classSymbol.declaredMembers().forEach(member -> parseSymbol(member, symbolNode, "member"));
      }
    } else if (symbol instanceof FunctionSymbol functionSymbol) {
      nodes.add(new GraphVisualizer.NodeBuilder(symbolNode)
        .addLabel("FunctionSymbol")
        .addLabel("name", functionSymbol.name())
        .addLabel("fullyQualifiedName", Optional.ofNullable(functionSymbol.fullyQualifiedName()).orElse("Ø"))
        .addLabel("parameters", Integer.toString(functionSymbol.parameters().size()))
        .addLabel("isAsynchronous", Boolean.toString(functionSymbol.isAsynchronous()))
        .addLabel("isInstanceMethod", Boolean.toString(functionSymbol.isInstanceMethod()))
        .addLabel("decorators", Integer.toString(functionSymbol.decorators().size()))
        .addLabel("hasDecorators", Boolean.toString(functionSymbol.hasDecorators()))
        .addLabel("isStub", Boolean.toString(functionSymbol.isStub()))
        .addLabel("annotatedReturnTypeName", Optional.ofNullable(functionSymbol.annotatedReturnTypeName()).orElse("Ø"))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "orange")
        .build());

      functionSymbol.parameters().forEach(parameter -> parseParameter(parameter, symbolNode, "parameter"));
    } else if (symbol instanceof AmbiguousSymbol ambiguousSymbol) {
      nodes.add(new GraphVisualizer.NodeBuilder(symbolNode)
        .addLabel("AmbiguousSymbol")
        .addLabel("alternatives", Integer.toString(ambiguousSymbol.alternatives().size()))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightcoral")
        .build());
      ambiguousSymbol.alternatives().forEach(alt -> parseSymbol(alt, symbolNode, "alternative"));
    } else if (symbol instanceof SymbolImpl symbolImpl) {
      nodes.add(new GraphVisualizer.NodeBuilder(symbolNode)
        .addLabel("SymbolImpl")
        .addLabel("name", symbolImpl.name())
        .addLabel("fullyQualifiedName", Optional.ofNullable(symbolImpl.fullyQualifiedName()).orElse("Ø"))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightblue")
        .build());
      parseType(symbolImpl.inferredType(), symbolNode, "inferredType");
    } else {
      throw new IllegalStateException("Unsupported symbol: " + symbol.getClass().getName());
    }
    // TODO: add SelfSymbolImpl (currently private)
    edges.add(new GraphVisualizer.Edge(currentNode, symbolNode, parentLabel));
  }

  private void parseParameter(FunctionSymbol.Parameter parameter, String symbolNode, String parentLabel) {
    String parameterNode = Integer.toString(System.identityHashCode(parameter));
    nodes.add(new GraphVisualizer.NodeBuilder(parameterNode)
      .addLabel("Parameter")
      .addLabel("name", parameter.name())
      .addLabel("hasDefaultValue", Boolean.toString(parameter.hasDefaultValue()))
      .addLabel("isKeywordVariadic", Boolean.toString(parameter.isKeywordVariadic()))
      .addLabel("isPositionalVariadic", Boolean.toString(parameter.isPositionalVariadic()))
      .addLabel("isKeywordOnly", Boolean.toString(parameter.isKeywordOnly()))
      .addLabel("isPositionalOnly", Boolean.toString(parameter.isPositionalOnly()))
      .extraProp("style", "filled")
      .extraProp("fillcolor", "green")
      .build());

    parseType(parameter.declaredType(), parameterNode, "inferredType");
    edges.add(new GraphVisualizer.Edge(symbolNode, parameterNode, parentLabel));
  }

  private void parse(String typeNode) {
    nodes.add(new GraphVisualizer.NodeBuilder(typeNode)
      .addLabel("AnyType")
      .extraProp("style", "filled")
      .extraProp("fillcolor", "green")
      .build());
  }

  private void parse(RuntimeType runtimeType, String typeNode) {
    try {
      var typeClassField = RuntimeType.class.getDeclaredField("typeClass");
      var builtinFullyQualifiedNameField = RuntimeType.class.getDeclaredField("builtinFullyQualifiedName");
      var typeClassSuperClassesFQNField = RuntimeType.class.getDeclaredField("typeClassSuperClassesFQN");
      typeClassField.setAccessible(true);
      builtinFullyQualifiedNameField.setAccessible(true);
      typeClassSuperClassesFQNField.setAccessible(true);
      nodes.add(new GraphVisualizer.NodeBuilder(typeNode)
        .addLabel("RuntimeType")
        .addLabel("gettypeClass()", runtimeType.getTypeClass().name())
        .addLabel("builtinFullyQualifiedName", Optional.ofNullable(builtinFullyQualifiedNameField.get(runtimeType)).orElse("Ø").toString())
        .addLabel("typeClassSuperClassesFQN",
          Optional.ofNullable(typeClassSuperClassesFQNField.get(runtimeType)).map(Set.class::cast).map(s -> (Stream<String>) s.stream())
            .map(s -> s.collect(Collectors.joining(","))).orElse("Ø"))
        .extraProp("style", "filled")
        .extraProp("fillcolor", "lightblue")
        .build());
      parseSymbol(runtimeType.getTypeClass(), typeNode, "typeClass");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
