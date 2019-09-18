/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.semantic;

import com.google.common.base.Functions;
import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.BeforeClass;
import org.junit.Test;
import org.sonar.python.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyLambdaExpressionTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.tree.BaseTreeVisitor;

import static org.assertj.core.api.Assertions.assertThat;

public class SymbolTableBuilderTreeTest {
  private static Map<String, PyFunctionDefTree> functionTreesByName = new HashMap<>();


  @BeforeClass
  public static void init() {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(new File("src/test/resources/semantic/symbols2.py"));
    PyFileInputTree fileInput = context.rootTree();
    fileInput.accept(new TestVisitor());
  }

  @Test
  public void local_variable() {
    PyFunctionDefTree functionTree = functionTreesByName.get("function_with_local");
    Set<TreeSymbol> symbols = functionTree.localVariables();
    Map<String, TreeSymbol> symbolByName = symbols.stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("a", "t2");
    TreeSymbol a = symbolByName.get("a");
    int functionStartLine = functionTree.firstToken().token().getLine();
    assertThat(a.usages()).extracting(tree -> tree.firstToken().token().getLine()).containsOnly(
      functionStartLine + 1, functionStartLine + 2, functionStartLine + 3, functionStartLine + 4);
    TreeSymbol t2 = symbolByName.get("t2");
    assertThat(t2.usages()).extracting(tree -> tree.firstToken().token().getLine()).containsOnly(
      functionStartLine + 5);
  }

  @Test
  public void free_variable() {
    PyFunctionDefTree functionTree = functionTreesByName.get("function_with_free_variable");
    assertThat(functionTree.localVariables()).isEmpty();
  }

  @Test
  public void rebound_variable() {
    PyFunctionDefTree functionTree = functionTreesByName.get("function_with_rebound_variable");
    Set<TreeSymbol> symbols = functionTree.localVariables();
    Map<String, TreeSymbol> symbolByName = symbols.stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("global_x");
  }

  @Test
  public void simple_parameter() {
    PyFunctionDefTree functionTree = functionTreesByName.get("simple_parameter");
    Map<String, TreeSymbol> symbolByName = functionTree.localVariables().stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("a");
  }

  @Test
  public void multiple_assignment() {
    PyFunctionDefTree functionTree = functionTreesByName.get("multiple_assignment");
    Set<TreeSymbol> symbols = functionTree.localVariables();
    Map<String, TreeSymbol> symbolByName = symbols.stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("x", "y");
    int functionStartLine = functionTree.firstToken().token().getLine();
    TreeSymbol x = symbolByName.get("x");
    assertThat(x.usages()).extracting(tree -> tree.firstToken().token().getLine()).containsOnly(functionStartLine + 1);
    TreeSymbol y = symbolByName.get("y");
    assertThat(y.usages()).extracting(tree -> tree.firstToken().token().getLine()).containsOnly(functionStartLine + 1);
  }

  @Test
  public void tuple_assignment() {
    PyFunctionDefTree functionTree = functionTreesByName.get("tuple_assignment");
    Set<TreeSymbol> symbols = functionTree.localVariables();
    Map<String, TreeSymbol> symbolByName = symbols.stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("x", "y");
    int functionStartLine = functionTree.firstToken().token().getLine();
    TreeSymbol x = symbolByName.get("x");
    assertThat(x.usages()).extracting(tree -> tree.firstToken().token().getLine()).containsOnly(functionStartLine + 1);
    TreeSymbol y = symbolByName.get("y");
    assertThat(y.usages()).extracting(tree -> tree.firstToken().token().getLine()).containsOnly(functionStartLine + 1);
  }

  @Test
  public void function_with_global_var() {
    PyFunctionDefTree functionTree = functionTreesByName.get("function_with_global_var");
    assertThat(functionTree.localVariables()).isEmpty();
  }

  @Test
  public void function_with_nonlocal_var() {
    PyFunctionDefTree functionTree = functionTreesByName.get("function_with_nonlocal_var");
    assertThat(functionTree.localVariables()).isEmpty();
  }

  @Test
  public void lambdas() {
    PyFunctionDefTree functionTree = functionTreesByName.get("function_with_lambdas");
    Map<String, TreeSymbol> symbolByName = functionTree.localVariables().stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));

    assertThat(symbolByName.keySet()).containsOnly("x", "y");
    TreeSymbol x = symbolByName.get("x");
    assertThat(x.usages()).hasSize(1);

    TreeSymbol y = symbolByName.get("y");
    assertThat(y.usages()).hasSize(2);

    List<PyLambdaExpressionTree> lambdas = functionTree.descendants(Tree.Kind.LAMBDA)
      .map(PyLambdaExpressionTree.class::cast)
      .collect(Collectors.toList());
    PyLambdaExpressionTree firstLambdaFunction = lambdas.get(0);
    symbolByName = firstLambdaFunction.localVariables().stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("x");
    TreeSymbol innerX = symbolByName.get("x");
    assertThat(innerX.usages()).hasSize(3);

    PyLambdaExpressionTree secondLambdaFunction = lambdas.get(1);
    symbolByName = secondLambdaFunction.localVariables().stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));
    assertThat(symbolByName.keySet()).containsOnly("z");
  }

  @Test
  public void for_stmt() {
    PyFunctionDefTree functionTree = functionTreesByName.get("function_with_loops");
    Map<String, TreeSymbol> symbolByName = functionTree.localVariables().stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));

    assertThat(symbolByName.keySet()).containsOnly("x");
    TreeSymbol x = symbolByName.get("x");
    assertThat(x.usages()).hasSize(1);
  }

  @Test
  public void comprehension() {
    PyFunctionDefTree functionTree = functionTreesByName.get("function_with_comprehension");
    Map<String, TreeSymbol> symbolByName = functionTree.localVariables().stream().collect(Collectors.toMap(TreeSymbol::name, Functions.identity()));

    assertThat(symbolByName.keySet()).containsOnly("a");
    TreeSymbol a = symbolByName.get("a");
    assertThat(a.usages()).hasSize(1);
  }

  private static class TestVisitor extends BaseTreeVisitor {
    @Override
    public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
      functionTreesByName.put(pyFunctionDefTree.name().name(), pyFunctionDefTree);
      super.visitFunctionDef(pyFunctionDefTree);
    }
  }

}
