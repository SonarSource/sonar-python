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

import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.impl.Parser;
import java.nio.charset.StandardCharsets;
import javax.annotation.Nullable;
import org.junit.Test;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;

public class FullyQualifiedNameTest {
  private Parser<Grammar> p = PythonParser.create(new PythonConfiguration(StandardCharsets.UTF_8));
  private PythonTreeMaker pythonTreeMaker = new PythonTreeMaker();
  @Test
  public void callee_qualified_expression() {
    PyFileInputTree tree = parse(
      "import mod",
      "mod.fn()"
    );
    assertNameAndQualifiedName(tree, "mod.fn", "mod.fn");
  }

  @Test
  public void callee_qualified_expression_without_import() {
    PyFileInputTree tree = parse(
      "mod.fn()"
    );
    assertNameAndQualifiedName(tree, "mod.fn", null);
  }

  @Test
  public void callee_name_without_import() {
    PyFileInputTree tree = parse(
      "fn()"
    );
    assertNameAndQualifiedName(tree, "fn", null);
  }

  @Test
  public void callee_subscription() {
    PyFileInputTree tree = parse(
      "foo['a']()"
    );
    PyCallExpressionTree callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNull();
  }

  @Test
  public void callee_qualified_expression_submodule() {
    PyFileInputTree tree = parse(
      "import mod.submod",
      "mod.submod.fn()"
    );
    assertNameAndQualifiedName(tree, "mod.submod.fn", "mod.submod.fn");
  }

  @Test
  public void var_callee_same_name_same_symbol() {
    PyFileInputTree tree = parse(
      "fn = 2",
      "fn('foo')"
    );
    PyCallExpressionTree callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.CALLEE);
    assertNameAndQualifiedName(tree, "fn", null);

    tree = parse(
      "fn = 2",
      "def foo():",
      "  fn('foo')"
    );
    callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.ASSIGNMENT_LHS, Usage.Kind.CALLEE);
  }

  @Test
  public void definition_callee_symbol() {
    PyFileInputTree tree = parse(
      "def fn(): pass",
      "fn('foo')"
    );
    PyCallExpressionTree callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    // TODO: create a symbol for function declaration and assert function decl and call expr symbols have the same reference
    assertThat(callExpression.calleeSymbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.CALLEE);
    assertNameAndQualifiedName(tree, "fn", null);
  }

  private void assertNameAndQualifiedName(PyFileInputTree tree, String name, @Nullable String qualifiedName) {
    PyCallExpressionTree callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNotNull();
    TreeSymbol symbol = callExpression.calleeSymbol();
    assertThat(symbol.name()).isEqualTo(name);
    assertThat(symbol.fullyQualifiedName()).isEqualTo(qualifiedName);
  }

  private PyFileInputTree parse(String ...lines) {
    String code = String.join(System.getProperty("line.separator"), lines);
    PyFileInputTree tree = pythonTreeMaker.fileInput(p.parse(code));
    new SymbolTableBuilder().visitFileInput(tree);
    return tree;
  }

}
