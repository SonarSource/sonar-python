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

import java.util.Set;
import org.apache.commons.lang.StringUtils;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;
import org.sonar.python.tree.AnnotatedAssignmentImpl;
import org.sonar.python.tree.AssignmentStatementImpl;
import org.sonar.python.types.v2.TypesTestUtils;
import org.sonar.python.types.visualization.visitors.ProjectLevelSymbolTableVisitor;
import org.sonar.python.types.visualization.visitors.TypeV1Visitor;
import org.sonar.python.types.visualization.visitors.TypeV2Visitor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.PythonTestUtils.pythonFile;

class GraphVisualizerTest {

  @Test
  void empty_graph() {
    var graphVisualizer = new GraphVisualizer();
    assertThat(graphVisualizer.nodes).isEmpty();
    assertThat(graphVisualizer.edges).isEmpty();
    assertThat(graphVisualizer).hasToString(
      """
        digraph G {
        node [
          shape = record
        ]
        graph [
          rankdir = "LR"
        ]
        }""");
  }

  @Test
  void simple_graph() {
    var graphVisualizer = new GraphVisualizer();
    graphVisualizer.nodes.add(new GraphVisualizer.Node("1", "Node1"));
    graphVisualizer.nodes.add(new GraphVisualizer.Node("2", "Node2"));
    graphVisualizer.edges.add(new GraphVisualizer.Edge("1", "2"));
    assertThat(graphVisualizer).hasToString(
      """
        digraph G {
        node [
          shape = record
        ]
        graph [
          rankdir = "LR"
        ]
        1 [label="Node1"]
        2 [label="Node2"]
        1 -> 2
        }""");
  }

  @Test
  void with_extra_properties() {
    var graphVisualizer = new GraphVisualizer();
    graphVisualizer.nodes.add(new GraphVisualizer.Node("1", "Node1", "color=\"red\", style=\"filled\", shape=\"box\""));
    graphVisualizer.nodes.add(new GraphVisualizer.Node("2", "Node2", "color=\"yellow\", style=\"filled\", shape=\"box\""));
    graphVisualizer.edges.add(new GraphVisualizer.Edge("1", "2", "parent"));
    assertThat(graphVisualizer).hasToString(
      """
        digraph G {
        node [
          shape = record
        ]
        graph [
          rankdir = "LR"
        ]
        1 [label="Node1", color="red", style="filled", shape="box"]
        2 [label="Node2", color="yellow", style="filled", shape="box"]
        1 -> 2 [label="parent"]
        }""");
  }

  @Test
  void builder_extra_properties() {
    var graphVisualizer = new GraphVisualizer();
    graphVisualizer.nodes.add(new GraphVisualizer.NodeBuilder("1")
      .addLabel("Node1")
      .build());
    graphVisualizer.nodes.add(new GraphVisualizer.NodeBuilder("2")
      .addLabel("Node2")
      .addLabel("something", "else")
      .color("yellow")
      .extraProp("style", "filled")
      .build());
    graphVisualizer.edges.add(new GraphVisualizer.Edge("1", "2", "parent"));
    assertThat(graphVisualizer).hasToString(
      """
        digraph G {
        node [
          shape = record
        ]
        graph [
          rankdir = "LR"
        ]
        1 [label="Node1"]
        2 [label="Node2 | {something | else}", color="yellow", style="filled"]
        1 -> 2 [label="parent"]
        }""");
  }

  private static class DummyCollector implements GraphVisualizer.TypeToGraphCollector<Object> {
    @Override
    public Set<GraphVisualizer.Edge> edges() {
      return Set.of();
    }

    @Override
    public Set<GraphVisualizer.Node> nodes() {
      return Set.of();
    }

    @Override
    public void parse(GraphVisualizer.Root<Object> root) {
    }
  }

  @Test
  void dummy_collectors() {
    var graphVisualizerBuilder = new GraphVisualizer.Builder();
    graphVisualizerBuilder.addCollector(new DummyCollector(), new GraphVisualizer.Root<>(new Object(), ""));
    graphVisualizerBuilder.addCollector(new DummyCollector(), new GraphVisualizer.Root<>(new Object(), ""));
    var graphVisualizer = graphVisualizerBuilder.build();
    assertThat(graphVisualizer).hasToString(
      """
        digraph G {
        node [
          shape = record
        ]
        graph [
          rankdir = "LR"
        ]
        }""");
  }

  private final String code_1 = """
    a = 1
    a = "2"
    def a():
      ...
    a = 2
    b = a
    ...
    c = 42
    def d(foo: int): ...
    class E():
      def e_1(self): ...
    class F(E):
      def e_2(self, x: int): ...
    e = E()
    f = F()

    class G(): ...
    class G(E): ...
    g = G()
    h: E = g
    if cond:
      smth = 42
    else:
      smth = "42"
    i = smth
    """;

  @Test
  void test_type_inference_v1() {
    FileInput fileInput = inferTypes(code_1);

    var a_1 = ((AssignmentStatementImpl) fileInput.statements().statements().get(0)).lhsExpressions().get(0).expressions().get(0).type();
    var a_2 = ((AssignmentStatementImpl) fileInput.statements().statements().get(1)).lhsExpressions().get(0).expressions().get(0).type();
    var a_3 = ((FunctionDef) fileInput.statements().statements().get(2)).name().type();
    var a_4 = ((AssignmentStatementImpl) fileInput.statements().statements().get(3)).lhsExpressions().get(0).expressions().get(0).type();
    var b = ((AssignmentStatementImpl) fileInput.statements().statements().get(4)).lhsExpressions().get(0).expressions().get(0).type();
    var c = ((AssignmentStatementImpl) fileInput.statements().statements().get(6)).lhsExpressions().get(0).expressions().get(0).type();
    var d = ((FunctionDef) fileInput.statements().statements().get(7)).name().type();
    var e = ((AssignmentStatementImpl) fileInput.statements().statements().get(10)).lhsExpressions().get(0).expressions().get(0).type();
    var f = ((AssignmentStatementImpl) fileInput.statements().statements().get(11)).lhsExpressions().get(0).expressions().get(0).type();
    var g = ((AssignmentStatementImpl) fileInput.statements().statements().get(14)).lhsExpressions().get(0).expressions().get(0).type();
    var h = ((AnnotatedAssignmentImpl) fileInput.statements().statements().get(15)).assignedValue().type();
    var i = ((AssignmentStatementImpl) fileInput.statements().statements().get(17)).lhsExpressions().get(0).expressions().get(0).type();

    var typeToGraph = new GraphVisualizer.Builder()
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(a_1, "a"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(a_2, "a"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(a_3, "a"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(a_4, "a"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(b, "b"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(c, "c"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(d, "d"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(e, "e"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(f, "f"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(g, "g"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(h, "h"))
      .addCollector(new TypeV1Visitor(), new GraphVisualizer.Root<>(i, "i"))
      .build();

    String out = typeToGraph.toString();

    assertThat(out)
      .containsOnlyOnce("a [label=\"a\"")
      .containsOnlyOnce("b [label=\"b\"")
      .containsOnlyOnce("c [label=\"c\"")
      .containsOnlyOnce("d [label=\"d\"")
      .containsOnlyOnce("e [label=\"e\"")
      .containsOnlyOnce("f [label=\"f\"")
      .containsOnlyOnce("AnyType");
    assertThat(StringUtils.countMatches(out, "Parameter")).isEqualTo(3);
    assertThat(StringUtils.countMatches(out, "FunctionSymbol")).isEqualTo(2);
    assertThat(StringUtils.countMatches(out, "ClassSymbol")).isEqualTo(7);
    assertThat(StringUtils.countMatches(out, "RuntimeType")).isEqualTo(6);
  }

  @Test
  void test_type_inference_v2() {
    FileInput fileInput = inferTypes(code_1);

    var a_1 = ((AssignmentStatementImpl) fileInput.statements().statements().get(0)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var a_2 = ((AssignmentStatementImpl) fileInput.statements().statements().get(1)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var a_3 = ((FunctionDef) fileInput.statements().statements().get(2)).name().typeV2();
    var a_4 = ((AssignmentStatementImpl) fileInput.statements().statements().get(3)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var b = ((AssignmentStatementImpl) fileInput.statements().statements().get(4)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var c = ((AssignmentStatementImpl) fileInput.statements().statements().get(6)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var d = ((FunctionDef) fileInput.statements().statements().get(7)).name().typeV2();
    var e = ((AssignmentStatementImpl) fileInput.statements().statements().get(10)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var f = ((AssignmentStatementImpl) fileInput.statements().statements().get(11)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var g = ((AssignmentStatementImpl) fileInput.statements().statements().get(14)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var h = ((AnnotatedAssignmentImpl) fileInput.statements().statements().get(15)).assignedValue().typeV2();
    var i = ((AssignmentStatementImpl) fileInput.statements().statements().get(17)).lhsExpressions().get(0).expressions().get(0).typeV2();

    var d_symbol = ((FunctionDef) fileInput.statements().statements().get(7)).name().symbolV2();
    var a_symbol = ((FunctionDef) fileInput.statements().statements().get(2)).name().symbolV2();

    Integer branchLimit = 3;
    var graphVisualizer = new GraphVisualizer.Builder()
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(a_1, "a"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(a_2, "a"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(a_3, "a"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(a_4, "a"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(b, "b"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(c, "c"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(d, "d"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(e, "e"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(f, "f"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(g, "g"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(h, "h"))
      .addCollector(new TypeV2Visitor.V2TypeInferenceVisitor(false, branchLimit, null, true), new GraphVisualizer.Root<>(i, "i"))
      .addCollector(new TypeV2Visitor.V2SymbolVisitor(), new GraphVisualizer.Root<>(a_symbol, "a"))
      .addCollector(new TypeV2Visitor.V2SymbolVisitor(), new GraphVisualizer.Root<>(d_symbol, "d"))
      .build();

    String out = graphVisualizer.toString();

    assertThat(out)
      .containsOnlyOnce("a [label=\"a\"")
      .containsOnlyOnce("b [label=\"b\"")
      .containsOnlyOnce("c [label=\"c\"")
      .containsOnlyOnce("d [label=\"d\"")
      .containsOnlyOnce("e [label=\"e\"")
      .containsOnlyOnce("f [label=\"f\"")
      .containsOnlyOnce("UnknownType");
    assertThat(StringUtils.countMatches(out, "ObjectType")).isEqualTo(8);
    assertThat(StringUtils.countMatches(out, "ClassType")).isEqualTo(4);
    assertThat(StringUtils.countMatches(out, "FunctionType")).isEqualTo(4);
    assertThat(StringUtils.countMatches(out, "ParameterV2")).isEqualTo(4);
    assertThat(StringUtils.countMatches(out, "SymbolV2")).isEqualTo(2);
    assertThat(StringUtils.countMatches(out, "UsageV2")).isEqualTo(6);
  }

  static PythonFile pythonFile = PythonTestUtils.pythonFile("");

  private static FileInput inferTypes(String lines) {
    return inferTypes(lines, TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE);
  }

  private static FileInput inferTypes(String lines, ProjectLevelTypeTable projectLevelTypeTable) {
    FileInput root = PythonTestUtils.parse(lines);

    var symbolTable = new SymbolTableBuilderV2(root)
      .build();
    new TypeInferenceV2(projectLevelTypeTable, pythonFile, symbolTable).inferTypes(root);
    return root;
  }

  @Test
  void project_level_symbol_table() {
    var graphVisualizerBuilder = new GraphVisualizer.Builder();
    String[] foo = {
      """
        from bar import B
        class A(B):
          def my_A_method(param: A): ...
          def my_A_other_method(param: B): ...
        """};
    String[] bar = {
      """
        class B:
          def my_B_method(param: B): ...
          def my_B_other_method(param: A): ...
        def ambiguous(param: int): ...
        class ambiguous: ...
        ambiguous = 42
        """
    };

    ProjectLevelSymbolTable projectLevelSymbolTable = new ProjectLevelSymbolTable();
    projectLevelSymbolTable.addModule(parseWithoutSymbols(foo), "", pythonFile("foo.py"));
    projectLevelSymbolTable.addModule(parseWithoutSymbols(bar), "", pythonFile("bar.py"));

    graphVisualizerBuilder.addCollector(new ProjectLevelSymbolTableVisitor(), new GraphVisualizer.Root<>(projectLevelSymbolTable, "ProjectLevelSymbolTable"));
    var graphVisualizer = graphVisualizerBuilder.build();
    String out = graphVisualizer.toString();

    assertThat(StringUtils.countMatches(out, "Module: ")).isEqualTo(2);
    assertThat(StringUtils.countMatches(out, "ClassDescriptor")).isEqualTo(3);
    assertThat(StringUtils.countMatches(out, "FunctionDescriptor")).isEqualTo(5);
    assertThat(StringUtils.countMatches(out, "Parameter")).isEqualTo(5);
    assertThat(StringUtils.countMatches(out, "VariableDescriptor")).isEqualTo(1);
    assertThat(StringUtils.countMatches(out, "AmbiguousDescriptor")).isEqualTo(1);
  }
}
