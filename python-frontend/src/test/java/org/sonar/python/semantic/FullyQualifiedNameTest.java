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
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;
import static org.sonar.python.PythonTestUtils.getAllDescendant;
import static org.sonar.python.PythonTestUtils.getFirstChild;
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
    CallExpression callExpression = getCallExpression(tree);
    assertThat(callExpression.calleeSymbol()).isNull();
  }

  @Test
  public void callee_name_without_import() {
    FileInput tree = parse(
      "fn()"
    );
    CallExpression callExpression = getCallExpression(tree);
    assertThat(callExpression.calleeSymbol()).isNull();
  }

  @Test
  public void callee_subscription() {
    FileInput tree = parse(
      "foo['a']()"
    );
    CallExpression callExpression = getCallExpression(tree);
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
    CallExpression callExpression = getCallExpression(tree);
    assertThat(callExpression.calleeSymbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.OTHER);
    assertNameAndQualifiedName(tree, "fn", "mod.fn");
  }

  @Test
  public void function_definition_callee_symbol() {
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", "my_module.py"),
      "def fn(): pass",
      "fn('foo')"
    );
    assertNameAndQualifiedName(tree, "fn", "my_package.my_module.fn");
  }

  @Test
  public void class_definition_callee_symbol() {
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", "my_module.py"),
      "class A: pass",
      "A()"
    );
    assertNameAndQualifiedName(tree, "A", "my_package.my_module.A");
  }

  @Test
  public void method_definition_symbol() {
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", "my_module.py"),
      "class A:",
      "  def fn(): pass"
    );
    FunctionDef method = (FunctionDef) getAllDescendant(tree, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    Symbol symbol = method.name().symbol();
    assertThat(symbol.name()).isEqualTo("fn");
    assertThat(symbol.fullyQualifiedName()).isEqualTo("my_package.my_module.A.fn");
  }

  @Test
  public void method_definition_subclass_symbol() {
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", "my_module.py"),
      "class A:",
      "  class B:",
      "    def fn(): pass"
    );
    FunctionDef method = (FunctionDef) getAllDescendant(tree, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    Symbol symbol = method.name().symbol();
    assertThat(symbol.name()).isEqualTo("fn");
    assertThat(symbol.fullyQualifiedName()).isEqualTo("my_package.my_module.A.B.fn");
  }

  @Test
  public void subfunction_definition() {
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", "my_module.py"),
      "def fn():",
      "  def inner(): pass",
      "  inner()"
    );
    assertNameAndQualifiedName(tree, "inner", "my_package.my_module.fn.inner");
  }

  @Test
  public void relative_import_symbols() {
    FileInput tree = parse(
      new SymbolTableBuilder("my_package", "my_module.py"),
      "from . import b"
    );
    Name b = getFirstChild(tree, t -> t.is(Tree.Kind.NAME));
    assertThat(b.symbol().fullyQualifiedName()).isEqualTo("my_package.b");

    // no package
    tree = parse(
      new SymbolTableBuilder("", "my_module.py"),
      "from . import b"
    );
    b = getFirstChild(tree, t -> t.is(Tree.Kind.NAME));
    assertThat(b.symbol().fullyQualifiedName()).isEqualTo("b");

    // no package, init file
    tree = parse(
      new SymbolTableBuilder("", "__init__.py"),
      "from . import b"
    );
    b = getFirstChild(tree, t -> t.is(Tree.Kind.NAME));
    assertThat(b.symbol().fullyQualifiedName()).isEqualTo("b");

    // two levels up
    tree = parse(
      new SymbolTableBuilder("my_package", "my_module.py"),
      "from ..my_package import b"
    );
    b = getFirstChild(tree, t -> t.is(Tree.Kind.NAME) && ((Name) t).name().equals("b"));
    assertThat(b.symbol().fullyQualifiedName()).isEqualTo("my_package.b");

    tree = parse(
      new SymbolTableBuilder("my_package1.my_package2", "my_module.py"),
      "from ..other import b"
    );
    b = getFirstChild(tree, t -> t.is(Tree.Kind.NAME) && ((Name) t).name().equals("b"));
    assertThat(b.symbol().fullyQualifiedName()).isEqualTo("my_package1.other.b");

    // overflow packages hierarchy
    tree = parse(
      new SymbolTableBuilder("my_package", "my_module.py"),
      "from ...my_package import b"
    );
    b = getFirstChild(tree, t -> t.is(Tree.Kind.NAME) && ((Name) t).name().equals("b"));
    assertThat(b.symbol().fullyQualifiedName()).isNull();

    // no fully qualified module name
    tree = parse(
      new SymbolTableBuilder(),
      "from . import b"
    );
    b = getFirstChild(tree, t -> t.is(Tree.Kind.NAME));
    assertThat(b.symbol().fullyQualifiedName()).isNull();
  }

  @Test
  public void imported_symbol() {
    FileInput tree = parse(
      "import mod"
    );
    Name nameTree = getFirstChild(tree, t -> t.is(Tree.Kind.NAME));
    assertThat(nameTree.symbol().fullyQualifiedName()).isEqualTo("mod");
    assertThat(nameTree.usage().kind()).isEqualTo(Usage.Kind.IMPORT);

    tree = parse(
      "import mod.submod"
    );
    nameTree = getNameEqualTo(tree, "mod");
    assertThat(nameTree.symbol().fullyQualifiedName()).isEqualTo("mod");

    nameTree = getNameEqualTo(tree, "submod");
    assertThat(nameTree.symbol()).isNull();
  }

  @Test
  public void from_imported_symbol() {
    FileInput tree = parse(
      "from mod import fn"
    );
    Name nameTree = getNameEqualTo(tree, "mod");
    assertThat(nameTree.symbol()).isNull();

    nameTree = getNameEqualTo(tree, "fn");
    assertThat(nameTree.symbol().fullyQualifiedName()).isEqualTo("mod.fn");
  }

  @Test
  public void two_usages_callee_symbol() {
    FileInput tree = parse(
      "import mod",
      "mod.fn()",
      "mod.fn()"
    );
    CallExpression callExpression = getCallExpression(tree);
    assertThat(callExpression.calleeSymbol().usages()).extracting(Usage::kind).containsExactly(Usage.Kind.OTHER, Usage.Kind.OTHER);
  }

  private static Name getNameEqualTo(FileInput tree, String strName) {
    return getFirstChild(tree, t -> t.is(Tree.Kind.NAME) && ((Name) t).name().equals(strName));
  }

  private static CallExpression getCallExpression(FileInput tree) {
    return getFirstChild(tree, t -> t.is(Tree.Kind.CALL_EXPR));
  }

  private static QualifiedExpression getQualifiedExpression(FileInput tree) {
    return getFirstChild(tree, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
  }

  @Test
  public void qualified_expression_symbol() {
    FileInput tree = parse(
      "import mod",
      "mod.prop"
    );
    QualifiedExpression qualifiedExpression = getQualifiedExpression(tree);
    Symbol symbol = qualifiedExpression.symbol();
    assertThat(symbol).isNotNull();
    assertThat(symbol.name()).isEqualTo("prop");
    assertThat(symbol.fullyQualifiedName()).isEqualTo("mod.prop");
    assertThat(qualifiedExpression.usage().kind()).isEqualTo(Usage.Kind.OTHER);
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
    tree.statements().statements().stream()
      .filter(t -> t.is(Tree.Kind.EXPRESSION_STMT))
      .map(t -> ((ExpressionStatement) t).expressions().get(0))
      .filter(t -> t.is(Tree.Kind.CALL_EXPR))
      .forEach(
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

  @Test
  public void init_module_relative_import() {
    String code = String.join(System.getProperty("line.separator"), "from .. import fn", "fn()", "class A: pass");
    FileInput fileInput = new PythonTreeMaker().fileInput(PythonParser.create().parse(code));
    PythonFile pythonFile = Mockito.mock(PythonFile.class, "__init__.py");
    when(pythonFile.fileName()).thenReturn("__init__.py");
    PythonVisitorContext context = new PythonVisitorContext(fileInput, pythonFile, null, "foo.bar");
    fileInput = context.rootTree();
    CallExpression callExpression = (CallExpression) getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR)).get(0);
    assertThat(callExpression.calleeSymbol().fullyQualifiedName()).isEqualTo("foo.fn");
    ClassDef classDef = (ClassDef) getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CLASSDEF)).get(0);
    assertThat(classDef.name().symbol().fullyQualifiedName()).isEqualTo("foo.bar.A");
  }


  private void assertNameAndQualifiedName(FileInput tree, String name, @Nullable String qualifiedName) {
    CallExpression callExpression = getCallExpression(tree);
    assertThat(callExpression.calleeSymbol()).isNotNull();
    Symbol symbol = callExpression.calleeSymbol();
    assertThat(symbol.name()).isEqualTo(name);
    assertThat(symbol.fullyQualifiedName()).isEqualTo(qualifiedName);
  }

}
