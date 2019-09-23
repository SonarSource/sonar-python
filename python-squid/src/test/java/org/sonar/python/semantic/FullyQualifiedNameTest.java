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
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
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
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void callee_qualified_expression_alias() {
    PyFileInputTree tree = parse(
      "import mod as alias",
      "alias.fn()"
    );
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void import_alias_reassigned() {
    PyFileInputTree tree = parse(
      "if x:",
      "  import mod1 as alias",
      "else:",
      "  import mod2 as alias",
      "alias.fn()"
    );
    assertNameAndQualifiedName(tree, "fn", null);
  }

  @Test
  public void callee_qualified_expression_without_import() {
    PyFileInputTree tree = parse(
      "mod.fn()"
    );
    PyCallExpressionTree callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNull();
  }

  @Test
  public void callee_name_without_import() {
    PyFileInputTree tree = parse(
      "fn()"
    );
    PyCallExpressionTree callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNull();
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
    assertNameAndQualifiedName(tree, "fn", "mod.submod.fn");
  }

  @Test
  public void var_callee_same_name_different_symbol() {
    PyFileInputTree tree = parse(
      "import mod",
      "fn = 2",
      "mod.fn('foo')"
    );
    PyCallExpressionTree callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.OTHER);
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void definition_callee_symbol() {
    PyFileInputTree tree = parse(
      "def fn(): pass",
      "fn('foo')"
    );
    // TODO: create a symbol for function declaration
    PyCallExpressionTree callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNull();
  }

  @Test
  public void imported_symbol() {
    PyFileInputTree tree = parse(
      "import mod"
    );
    PyNameTree nameTree = (PyNameTree) tree.descendants(Tree.Kind.NAME).findFirst().get();
    assertThat(nameTree.symbol().fullyQualifiedName()).isEqualTo("mod");

    tree = parse(
      "import mod.submod"
    );
    nameTree = (PyNameTree) tree.descendants(Tree.Kind.NAME)
      .filter(name -> ((PyNameTree) name).name().equals("mod"))
      .findFirst().get();
    assertThat(nameTree.symbol().fullyQualifiedName()).isEqualTo("mod");

    nameTree = (PyNameTree) tree.descendants(Tree.Kind.NAME)
      .filter(name -> ((PyNameTree) name).name().equals("submod"))
      .findFirst().get();
    assertThat(nameTree.symbol()).isNull();
  }

  @Test
  public void from_imported_symbol() {
    PyFileInputTree tree = parse(
      "from mod import fn"
    );
    PyNameTree nameTree = (PyNameTree) tree.descendants(Tree.Kind.NAME)
      .filter(name -> ((PyNameTree) name).name().equals("mod"))
      .findFirst().get();
    assertThat(nameTree.symbol()).isNull();

    nameTree = (PyNameTree) tree.descendants(Tree.Kind.NAME)
      .filter(name -> ((PyNameTree) name).name().equals("fn"))
      .findFirst().get();
    assertThat(nameTree.symbol().fullyQualifiedName()).isEqualTo("mod.fn");
  }

  @Test
  public void two_usages_callee_symbol() {
    PyFileInputTree tree = parse(
      "import mod",
      "mod.fn()",
      "mod.fn()"
    );
    PyCallExpressionTree callExpression = (PyCallExpressionTree) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.OTHER, Usage.Kind.OTHER);
  }

  @Test
  public void qualified_expression_symbol() {
    PyFileInputTree tree = parse(
      "import mod",
      "mod.prop"
    );
    PyQualifiedExpressionTree qualifiedExpression = (PyQualifiedExpressionTree) tree.descendants(Tree.Kind.QUALIFIED_EXPR).findFirst().get();
    TreeSymbol symbol = qualifiedExpression.symbol();
    assertThat(symbol).isNotNull();
    assertThat(symbol.name()).isEqualTo("prop");
    assertThat(symbol.fullyQualifiedName()).isEqualTo("mod.prop");
  }

  @Test
  public void from_import() {
    PyFileInputTree tree = parse(
      "from mod import fn",
      "fn('foo')"
    );
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void from_import_multiple_names() {
    PyFileInputTree tree = parse(
      "from mod import f, g, h",
      "f('foo')",
      "g('foo')",
      "h('foo')"
    );
    tree.descendants(Tree.Kind.CALL_EXPR).forEach(
      descendant -> {
        PyCallExpressionTree callExpression = (PyCallExpressionTree) descendant;
        assertThat(callExpression.calleeSymbol()).isNotNull();
        TreeSymbol symbol = callExpression.calleeSymbol();
        PyNameTree callee = (PyNameTree) callExpression.callee();
        assertThat(symbol.fullyQualifiedName()).isEqualTo("mod." + callee.name());
      }
    );
  }

  @Test
  public void from_import_reassigned() {
    PyFileInputTree tree = parse(
      "fn = 42",
      "from mod import fn",
      "fn('foo')"
    );
    assertNameAndQualifiedName(tree, "fn", null);

    tree = parse(
      "from mod import fn",
      "fn = 42",
      "fn('foo')"
    );
    assertNameAndQualifiedName(tree, "fn", null);
  }

  @Test
  public void from_import_alias() {
    PyFileInputTree tree = parse(
      "from mod import fn as g",
      "g('foo')"
    );
    assertNameAndQualifiedName(tree, "g", "mod.fn");
  }

  @Test
  public void from_import_relative() {
    PyFileInputTree tree = parse(
      "from . import fn",
      "fn('foo')"
    );
    assertNameAndQualifiedName(tree, "fn", null);

    tree = parse(
      "from .foo import fn",
      "fn('foo')"
    );
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
