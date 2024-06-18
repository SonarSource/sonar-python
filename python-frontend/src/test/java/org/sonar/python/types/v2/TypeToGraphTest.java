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
package org.sonar.python.types.v2;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;
import org.sonar.python.tree.AssignmentStatementImpl;

import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;

class TypeToGraphTest {

  static PythonFile pythonFile = PythonTestUtils.pythonFile("");

  private static FileInput inferTypes(String lines) {
    return inferTypes(lines, PROJECT_LEVEL_TYPE_TABLE);
  }

  private static FileInput inferTypes(String lines, ProjectLevelTypeTable projectLevelTypeTable) {
    FileInput root = parse(lines);

    var symbolTable = new SymbolTableBuilderV2(root)
      .build();
    new TypeInferenceV2(projectLevelTypeTable, pythonFile, symbolTable).inferTypes(root);
    return root;
  }

  @Test
  void test_type_inference_v2() {
    String input = """
      a = 1
      a = "2"
      def a():
      	...
      a = 2
      b = a
      ...
      c = 42
      def d(): ...
      class E():
        def e_1(self): ...
      class F(E):
        def e_2(self): ...
      e = E()
      f = F()
      """;

    FileInput fileInput = inferTypes(input);

    var a_1 = ((AssignmentStatementImpl) fileInput.statements().statements().get(0)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var a_2 = ((AssignmentStatementImpl) fileInput.statements().statements().get(1)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var a_3 = ((FunctionDef) fileInput.statements().statements().get(2)).name().typeV2();
    var a_4 = ((AssignmentStatementImpl) fileInput.statements().statements().get(3)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var b = ((AssignmentStatementImpl) fileInput.statements().statements().get(4)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var c = ((AssignmentStatementImpl) fileInput.statements().statements().get(6)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var d = ((FunctionDef) fileInput.statements().statements().get(7)).name().typeV2();
    var e = ((AssignmentStatementImpl) fileInput.statements().statements().get(10)).lhsExpressions().get(0).expressions().get(0).typeV2();
    var f = ((AssignmentStatementImpl) fileInput.statements().statements().get(11)).lhsExpressions().get(0).expressions().get(0).typeV2();

    var d_symbol = ((FunctionDef) fileInput.statements().statements().get(7)).name().symbolV2();
    var a_symbol = ((FunctionDef) fileInput.statements().statements().get(2)).name().symbolV2();

    Integer branchLimit = 3;
    var typeToGraph = new TypeToGraph.Builder()
      .addCollector(new TypeToGraph.V2TypeInferenceVisitor(false, branchLimit, null, true), new TypeToGraph.Root<>(a_1, "a"))
      .addCollector(new TypeToGraph.V2TypeInferenceVisitor(false, branchLimit, null, true), new TypeToGraph.Root<>(a_2, "a"))
      .addCollector(new TypeToGraph.V2TypeInferenceVisitor(false, branchLimit, null, true), new TypeToGraph.Root<>(a_3, "a"))
      .addCollector(new TypeToGraph.V2TypeInferenceVisitor(false, branchLimit, null, true), new TypeToGraph.Root<>(a_4, "a"))
      .addCollector(new TypeToGraph.V2TypeInferenceVisitor(false, branchLimit, null, true), new TypeToGraph.Root<>(b, "b"))
      .addCollector(new TypeToGraph.V2TypeInferenceVisitor(false, branchLimit, null, true), new TypeToGraph.Root<>(c, "c"))
      .addCollector(new TypeToGraph.V2TypeInferenceVisitor(false, branchLimit, null, true), new TypeToGraph.Root<>(d, "d"))
      .addCollector(new TypeToGraph.V2TypeInferenceVisitor(false, branchLimit, null, true), new TypeToGraph.Root<>(e, "e"))
      .addCollector(new TypeToGraph.V2TypeInferenceVisitor(false, branchLimit, null, true), new TypeToGraph.Root<>(f, "f"))

      .addCollector(new TypeToGraph.V2SymbolVisitor(), new TypeToGraph.Root<>(a_symbol, "a"))
      .addCollector(new TypeToGraph.V2SymbolVisitor(), new TypeToGraph.Root<>(d_symbol, "d"))
      .build();

    String out = typeToGraph.toString();
    System.out.println(out);
  }
  @Test
  void test_type_inference_v1() {
    String input = """
      a = 1
      a = "2"
      def a():
      	...
      a = 2
      b = a
      ...
      c = 42
      def d(): ...
      class E():
        def e_1(self, x: int) -> float: ...
      class F(E):
        def e_2(self): ...
      e = E()
      f = F()
      """;

    FileInput fileInput = inferTypes(input);

    var a_1 = ((AssignmentStatementImpl) fileInput.statements().statements().get(0)).lhsExpressions().get(0).expressions().get(0).type();
    var a_2 = ((AssignmentStatementImpl) fileInput.statements().statements().get(1)).lhsExpressions().get(0).expressions().get(0).type();
    var a_3 = ((FunctionDef) fileInput.statements().statements().get(2)).name().type();
    var a_4 = ((AssignmentStatementImpl) fileInput.statements().statements().get(3)).lhsExpressions().get(0).expressions().get(0).type();
    var b = ((AssignmentStatementImpl) fileInput.statements().statements().get(4)).lhsExpressions().get(0).expressions().get(0).type();
    var c = ((AssignmentStatementImpl) fileInput.statements().statements().get(6)).lhsExpressions().get(0).expressions().get(0).type();
    var d = ((FunctionDef) fileInput.statements().statements().get(7)).name().type();
    var e = ((AssignmentStatementImpl) fileInput.statements().statements().get(10)).lhsExpressions().get(0).expressions().get(0).type();
    var f = ((AssignmentStatementImpl) fileInput.statements().statements().get(11)).lhsExpressions().get(0).expressions().get(0).type();



    Integer branchLimit = null;
    var typeToGraph = new TypeToGraph.Builder()
      .addCollector(new TypeToGraph.TypeV1Visitor(), new TypeToGraph.Root<>(a_1, "a"))
      .addCollector(new TypeToGraph.TypeV1Visitor(), new TypeToGraph.Root<>(a_2, "a"))
      .addCollector(new TypeToGraph.TypeV1Visitor(), new TypeToGraph.Root<>(a_3, "a"))
      .addCollector(new TypeToGraph.TypeV1Visitor(), new TypeToGraph.Root<>(a_4, "a"))
      .addCollector(new TypeToGraph.TypeV1Visitor(), new TypeToGraph.Root<>(b, "b"))
      .addCollector(new TypeToGraph.TypeV1Visitor(), new TypeToGraph.Root<>(c, "c"))
      .addCollector(new TypeToGraph.TypeV1Visitor(), new TypeToGraph.Root<>(d, "d"))
      .addCollector(new TypeToGraph.TypeV1Visitor(), new TypeToGraph.Root<>(e, "e"))
      .addCollector(new TypeToGraph.TypeV1Visitor(), new TypeToGraph.Root<>(f, "f"))
      .build();

    String out = typeToGraph.toString();
    System.out.println(out);
  }
}
