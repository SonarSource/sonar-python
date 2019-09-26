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

import javax.annotation.Nullable;
import org.junit.Test;
import org.sonar.python.api.tree.CallExpression;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.QualifiedExpression;
import org.sonar.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;

public class FullyQualifiedNameTest {

  @Test
  public void callee_qualified_expression() {
    FileInput tree = parse(
      "import mod",
      "mod.fn()"
    );
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void callee_qualified_expression_alias() {
    FileInput tree = parse(
      "import mod as alias",
      "alias.fn()"
    );
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void import_alias_reassigned() {
    FileInput tree = parse(
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
    FileInput tree = parse(
      "mod.fn()"
    );
    CallExpression callExpression = (CallExpression) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNull();
  }

  @Test
  public void callee_name_without_import() {
    FileInput tree = parse(
      "fn()"
    );
    CallExpression callExpression = (CallExpression) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNull();
  }

  @Test
  public void callee_subscription() {
    FileInput tree = parse(
      "foo['a']()"
    );
    CallExpression callExpression = (CallExpression) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNull();
  }

  @Test
  public void callee_qualified_expression_submodule() {
    FileInput tree = parse(
      "import mod.submod",
      "mod.submod.fn()"
    );
    assertNameAndQualifiedName(tree, "fn", "mod.submod.fn");
  }

  @Test
  public void var_callee_same_name_different_symbol() {
    FileInput tree = parse(
      "import mod",
      "fn = 2",
      "mod.fn('foo')"
    );
    CallExpression callExpression = (CallExpression) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.OTHER);
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void definition_callee_symbol() {
    FileInput tree = parse(
      "def fn(): pass",
      "fn('foo')"
    );
    // TODO: create a symbol for function declaration
    CallExpression callExpression = (CallExpression) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNull();
  }

  @Test
  public void imported_symbol() {
    FileInput tree = parse(
      "import mod"
    );
    Name nameTree = (Name) tree.descendants(Tree.Kind.NAME).findFirst().get();
    assertThat(nameTree.symbol().fullyQualifiedName()).isEqualTo("mod");

    tree = parse(
      "import mod.submod"
    );
    nameTree = (Name) tree.descendants(Tree.Kind.NAME)
      .filter(name -> ((Name) name).name().equals("mod"))
      .findFirst().get();
    assertThat(nameTree.symbol().fullyQualifiedName()).isEqualTo("mod");

    nameTree = (Name) tree.descendants(Tree.Kind.NAME)
      .filter(name -> ((Name) name).name().equals("submod"))
      .findFirst().get();
    assertThat(nameTree.symbol()).isNull();
  }

  @Test
  public void from_imported_symbol() {
    FileInput tree = parse(
      "from mod import fn"
    );
    Name nameTree = (Name) tree.descendants(Tree.Kind.NAME)
      .filter(name -> ((Name) name).name().equals("mod"))
      .findFirst().get();
    assertThat(nameTree.symbol()).isNull();

    nameTree = (Name) tree.descendants(Tree.Kind.NAME)
      .filter(name -> ((Name) name).name().equals("fn"))
      .findFirst().get();
    assertThat(nameTree.symbol().fullyQualifiedName()).isEqualTo("mod.fn");
  }

  @Test
  public void two_usages_callee_symbol() {
    FileInput tree = parse(
      "import mod",
      "mod.fn()",
      "mod.fn()"
    );
    CallExpression callExpression = (CallExpression) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.OTHER, Usage.Kind.OTHER);
  }

  @Test
  public void qualified_expression_symbol() {
    FileInput tree = parse(
      "import mod",
      "mod.prop"
    );
    QualifiedExpression qualifiedExpression = (QualifiedExpression) tree.descendants(Tree.Kind.QUALIFIED_EXPR).findFirst().get();
    Symbol symbol = qualifiedExpression.symbol();
    assertThat(symbol).isNotNull();
    assertThat(symbol.name()).isEqualTo("prop");
    assertThat(symbol.fullyQualifiedName()).isEqualTo("mod.prop");
  }

  @Test
  public void from_import() {
    FileInput tree = parse(
      "from mod import fn",
      "fn('foo')"
    );
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void from_import_multiple_names() {
    FileInput tree = parse(
      "from mod import f, g, h",
      "f('foo')",
      "g('foo')",
      "h('foo')"
    );
    tree.descendants(Tree.Kind.CALL_EXPR).forEach(
      descendant -> {
        CallExpression callExpression = (CallExpression) descendant;
        assertThat(callExpression.calleeSymbol()).isNotNull();
        Symbol symbol = callExpression.calleeSymbol();
        Name callee = (Name) callExpression.callee();
        assertThat(symbol.fullyQualifiedName()).isEqualTo("mod." + callee.name());
      }
    );
  }

  @Test
  public void from_import_reassigned() {
    FileInput tree = parse(
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
  public void import_reassigned_exceptions() {
    FileInput tree = parse(
      "import mod",
      "import mod",
      "mod.fn('foo')"
    );
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void from_import_alias() {
    FileInput tree = parse(
      "from mod import fn as g",
      "g('foo')"
    );
    assertNameAndQualifiedName(tree, "g", "mod.fn");
  }

  @Test
  public void from_import_relative() {
    FileInput tree = parse(
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

  @Test
  public void from_import_submodule() {
    FileInput tree = parse(
      "from mod.submod import fn",
      "fn('foo')"
    );
    assertNameAndQualifiedName(tree, "fn", "mod.submod.fn");

    tree = parse(
      "from mod import submod",
      "submod.fn('foo')"
    );
    assertNameAndQualifiedName(tree, "fn", "mod.submod.fn");
  }


  private void assertNameAndQualifiedName(FileInput tree, String name, @Nullable String qualifiedName) {
    CallExpression callExpression = (CallExpression) tree.descendants(Tree.Kind.CALL_EXPR).findFirst().get();
    assertThat(callExpression.calleeSymbol()).isNotNull();
    Symbol symbol = callExpression.calleeSymbol();
    assertThat(symbol.name()).isEqualTo(name);
    assertThat(symbol.fullyQualifiedName()).isEqualTo(qualifiedName);
  }

}
