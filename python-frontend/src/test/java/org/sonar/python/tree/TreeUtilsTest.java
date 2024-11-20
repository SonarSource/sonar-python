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
package org.sonar.python.tree;

import com.sonar.sslr.api.AstNode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.PassStatement;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.SymbolTableBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.pythonFile;

class TreeUtilsTest {

  @Test
  void first_ancestor_of_kind() {
    String code = "" +
      "class A:\n" +
      "  def foo(): pass";
    FileInput root = parse(code);
    assertThat(TreeUtils.firstAncestorOfKind(root, Kind.CLASSDEF)).isNull();
    ClassDef classDef = (ClassDef) root.statements().statements().get(0);
    assertThat(TreeUtils.firstAncestorOfKind(classDef, Kind.FILE_INPUT, Kind.CLASSDEF)).isEqualTo(root);
    FunctionDef funcDef = (FunctionDef) classDef.body().statements().get(0);
    assertThat(TreeUtils.firstAncestorOfKind(funcDef, Kind.FILE_INPUT)).isEqualTo(root);
    assertThat(TreeUtils.firstAncestorOfKind(funcDef, Kind.CLASSDEF)).isEqualTo(classDef);

    code = "" +
      "while True:\n" +
      "  while True:\n" +
      "    pass";
    WhileStatement outerWhile = (WhileStatement) parse(code).statements().statements().get(0);
    WhileStatement innerWhile = (WhileStatement) outerWhile.body().statements().get(0);
    PassStatement passStatement = (PassStatement) innerWhile.body().statements().get(0);
    assertThat(TreeUtils.firstAncestorOfKind(passStatement, Kind.WHILE_STMT)).isEqualTo(innerWhile);
  }

  @Test
  void first_ancestor() {
    String code = "" +
      "def outer():\n" +
      "  def inner():\n" +
      "    pass";
    FileInput root = parse(code);
    FunctionDef outerFunction = (FunctionDef) root.statements().statements().get(0);
    FunctionDef innerFunction = (FunctionDef) outerFunction.body().statements().get(0);
    Statement passStatement = innerFunction.body().statements().get(0);
    assertThat(TreeUtils.firstAncestor(passStatement, TreeUtilsTest::isOuterFunction)).isEqualTo(outerFunction);
  }

  @Test
  void tokens() {
    // simple statement parsed so that we easily get all tokens from members or first token.
    FileInput parsed = parse("if foo:\n  pass");
    IfStatement ifStmt = (IfStatement) parsed.statements().statements().get(0);
    List<Token> collect = new ArrayList<>(ifStmt.children().stream().map(t -> t.is(Kind.TOKEN) ? (Token) t : t.firstToken()).toList());
    collect.add(parsed.lastToken());
    assertThat(TreeUtils.tokens(parsed)).containsExactly(collect.toArray(new Token[0]));

    assertThat(TreeUtils.tokens(parsed.lastToken())).containsExactly(parsed.lastToken());

  }

  @Test
  void non_whitespace_tokens() {
    FileInput parsed = parse("if foo:\n  pass");
    IfStatement ifStmt = (IfStatement) parsed.statements().statements().get(0);
    List<Token> nonWhitespaceTokens = TreeUtils.nonWhitespaceTokens(ifStmt);
    nonWhitespaceTokens.forEach(t -> assertThat(t.type()).isNotIn(PythonTokenType.NEWLINE, PythonTokenType.INDENT, PythonTokenType.DEDENT));
    assertThat(nonWhitespaceTokens).hasSize(4);
    assertThat(nonWhitespaceTokens.stream().map(Token::value)).containsExactly("if", "foo", ":", "pass");
  }

  @Test
  void hasDescendants() {
    FileInput fileInput = parse("class A:\n  def foo(): pass");
    assertThat(TreeUtils.hasDescendant(fileInput, t -> t.is(Kind.PASS_STMT))).isTrue();
    assertThat(TreeUtils.hasDescendant(fileInput, t -> (t.is(Kind.NAME) && ((Name) t).name().equals("foo")))).isTrue();
    assertThat(TreeUtils.hasDescendant(fileInput, t -> (t.is(Kind.NAME) && ((Name) t).name().equals("bar")))).isFalse();
    assertThat(TreeUtils.hasDescendant(fileInput, t -> t.is(Kind.IF_STMT))).isFalse();
  }

  @Test
  void getSymbolFromTree() {
    assertThat(TreeUtils.getSymbolFromTree(null)).isEmpty();

    Expression expression = lastExpression(
      "x = 42",
      "x");
    assertThat(TreeUtils.getSymbolFromTree(expression)).contains(((HasSymbol) expression).symbol());

    expression = lastExpression("foo()");
    assertThat(TreeUtils.getSymbolFromTree(expression)).isEmpty();
  }

  @Test
  void getClassSymbolFromDef() {
    FileInput fileInput = PythonTestUtils.parse("class A:\n  def foo(): pass");
    ClassDef classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));

    Symbol symbolA = classDef.name().symbol();
    assertThat(TreeUtils.getClassSymbolFromDef(classDef)).isEqualTo(symbolA);
    assertThat(TreeUtils.getClassSymbolFromDef(null)).isNull();

    fileInput = PythonTestUtils.parse(
      "class A:",
      "    pass",
      "A = 42");
    classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));
    assertThat(TreeUtils.getClassSymbolFromDef(classDef)).isNull();
  }

  @Test
  void test_getParentClassesFQN() {
    String code = "class A:\n  def foo(): pass";
    FileInput fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    ClassDef classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));
    assertThat(TreeUtils.getParentClassesFQN(classDef)).isEmpty();

    code = "class B: ...\nclass A(B):\n  def foo(): pass";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));
    assertThat(TreeUtils.getParentClassesFQN(classDef)).containsExactlyInAnyOrder("mod1.B");

    code = "class B: ...\nclass C: ...\nclass A(B,C):\n  def foo(): pass";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));
    assertThat(TreeUtils.getParentClassesFQN(classDef)).containsExactlyInAnyOrder("mod1.B", "mod1.C");

    code = "class B: ...\nclass C(B): ...\nclass A(C):\n  def foo(): pass";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));
    assertThat(TreeUtils.getParentClassesFQN(classDef)).containsExactlyInAnyOrder("mod1.B", "mod1.C");

    code = "if cond:\n  class A:...\nelse:\n  class A: pass";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));
    assertThat(TreeUtils.getParentClassesFQN(classDef)).isEmpty();

    code = "import b \nclass A(b.B): ...";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));
    assertThat(TreeUtils.getParentClassesFQN(classDef)).containsExactly("b.B");
  }

  @Test
  void getClassSymbolFromDef_illegalSymbol() {
    FileInput fileInput = PythonTestUtils.parseWithoutSymbols("class A:\n  def foo(): pass");
    ClassDef classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));

    assertThatThrownBy(() -> TreeUtils.getClassSymbolFromDef(classDef)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void getFunctionSymbolFromDef() {
    FileInput fileInput = PythonTestUtils.parse("def foo(): pass");
    FunctionDef functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));

    Symbol symbolFoo = functionDef.name().symbol();
    assertThat(TreeUtils.getFunctionSymbolFromDef(functionDef)).isEqualTo(symbolFoo);
    assertThat(TreeUtils.getFunctionSymbolFromDef(null)).isNull();

    fileInput = PythonTestUtils.parse(
      "def foo():",
      "    pass",
      "foo = 42");
    functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.getFunctionSymbolFromDef(functionDef)).isNull();
  }

  @Test
  void getFunctionSymbolFromDef_illegalSymbol() {
    FileInput fileInput = PythonTestUtils.parseWithoutSymbols("def foo(): pass");
    FunctionDef functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));

    assertThatThrownBy(() -> TreeUtils.getFunctionSymbolFromDef(functionDef)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void nonTupleParameters() {
    FileInput fileInput = PythonTestUtils.parse("def foo(): pass");
    FunctionDef functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.nonTupleParameters(functionDef)).isEmpty();

    fileInput = PythonTestUtils.parse("def foo(param1, param2): pass");
    functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.nonTupleParameters(functionDef)).isEqualTo(functionDef.parameters().nonTuple());
  }

  @Test
  void positionalParameters() {
    FileInput fileInput = PythonTestUtils.parse("def foo(): pass");
    FunctionDef functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.positionalParameters(functionDef)).isEmpty();

    fileInput = PythonTestUtils.parse("def foo(param1, param2): pass");
    functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.positionalParameters(functionDef)).isEqualTo(functionDef.parameters().all());

    fileInput = PythonTestUtils.parse("def foo(param1, *param2): pass");
    functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.positionalParameters(functionDef)).isEqualTo(functionDef.parameters().all());

    fileInput = PythonTestUtils.parse("def foo(param1, param2, *, kw1, kw2): pass");
    functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.positionalParameters(functionDef)).isEqualTo(functionDef.parameters().all().subList(0, 2));

    fileInput = PythonTestUtils.parse("def foo((param1, param2), *, kw1, kw2): pass");
    functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.positionalParameters(functionDef)).isEmpty();

    fileInput = PythonTestUtils.parse("def foo(param1, /, param2, *, kw1, kw2): pass");
    functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    List<AnyParameter> parameters = functionDef.parameters().all();
    assertThat(TreeUtils.positionalParameters(functionDef)).isEqualTo(
      Arrays.asList(parameters.get(0), parameters.get(2)));
  }

  @Test
  void topLevelFunctionDefs() {
    FileInput fileInput = PythonTestUtils.parse(
      "class A:",
      "    x = True",
      "    def foo(self): pass",
      "    if x:",
      "        def bar(self, x): return 1",
      "    else:",
      "        def baz(self, x, y): return x + y");

    ClassDef classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));
    List<FunctionDef> functionDefs = PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Kind.FUNCDEF));

    assertThat(TreeUtils.topLevelFunctionDefs(classDef)).containsAll(functionDefs);

    fileInput = PythonTestUtils.parse(
      "class A:",
      "    x = True",
      "    def foo(self):",
      "        def foo2(x, y): return x + y",
      "        return foo2(1, 1)",
      "    class B:",
      "        def bar(self): pass");
    classDef = PythonTestUtils.getFirstChild(fileInput, t -> t.is(Kind.CLASSDEF));
    FunctionDef fooDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF) && ((FunctionDef) t).name().name().equals("foo"));

    assertThat(TreeUtils.topLevelFunctionDefs(classDef)).isEqualTo(Collections.singletonList(fooDef));
  }

  @Test
  void test_nthArgumentOrKeyword() {
    FileInput fileInput = PythonTestUtils.parse(
      "def foo(p0, p1, p2): ...",
      "foo(1, 2, p2 = 3)");
    CallExpression callExpr = PythonTestUtils.getLastDescendant(fileInput, tree -> tree.is(Kind.CALL_EXPR));
    RegularArgument p1 = TreeUtils.nthArgumentOrKeyword(1, "p1", callExpr.arguments());
    assertThat(p1.expression().is(Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(((NumericLiteral) p1.expression()).valueAsLong()).isEqualTo(2);

    RegularArgument p2 = TreeUtils.nthArgumentOrKeyword(2, "p2", callExpr.arguments());
    assertThat(p2.expression().is(Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(((NumericLiteral) p2.expression()).valueAsLong()).isEqualTo(3);
  }

  @Test
  void test_nthArgumentOrKeyword_unpacking() {
    FileInput fileInput = PythonTestUtils.parse(
      "def foo(p0, p1, p2): ...",
      "args = [1, 2, 3]",
      "foo(*args)");

    CallExpression callExpr = PythonTestUtils.getLastDescendant(fileInput, tree -> tree.is(Kind.CALL_EXPR));
    RegularArgument p1 = TreeUtils.nthArgumentOrKeyword(1, "p1", callExpr.arguments());
    assertThat(p1).isNull();
  }

  @Test
  void test_nthArgumentOrKeyword_no_positional() {
    FileInput fileInput = PythonTestUtils.parse(
      "def foo(p0, p1 = 2, p2 = 3): ...",
      "foo(0, p2 = 4)");

    CallExpression callExpr = PythonTestUtils.getLastDescendant(fileInput, tree -> tree.is(Kind.CALL_EXPR));
    RegularArgument p1 = TreeUtils.nthArgumentOrKeyword(1, "p1", callExpr.arguments());
    assertThat(p1).isNull();
  }

  @Test
  void test_argumentByKeyword() {
    FileInput fileInput = PythonTestUtils.parse(
      "def foo(p0, p1, p2): ...",
      "foo(p1 = 1, p2 = 2)");
    CallExpression callExpr = PythonTestUtils.getLastDescendant(fileInput, tree -> tree.is(Kind.CALL_EXPR));
    RegularArgument p1 = TreeUtils.argumentByKeyword("p1", callExpr.arguments());
    assertThat(p1.expression().is(Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(((NumericLiteral) p1.expression()).valueAsLong()).isEqualTo(1);

    RegularArgument p2 = TreeUtils.argumentByKeyword("p2", callExpr.arguments());
    assertThat(p2.expression().is(Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(((NumericLiteral) p2.expression()).valueAsLong()).isEqualTo(2);

    RegularArgument p3 = TreeUtils.argumentByKeyword("p3", callExpr.arguments());
    assertThat(p3).isNull();
  }

  @Test
  void test_isBooleanLiteral() {
    assertThat(TreeUtils.isBooleanLiteral(lastExpression("True"))).isTrue();
    assertThat(TreeUtils.isBooleanLiteral(lastExpression("False"))).isTrue();
    assertThat(TreeUtils.isBooleanLiteral(lastExpression("x"))).isFalse();
    assertThat(TreeUtils.isBooleanLiteral(lastExpression("foo()"))).isFalse();
  }

  @Test
  void test_nameFromExpression() {
    assertThat(TreeUtils.nameFromExpression(lastExpression("my_var"))).isEqualTo("my_var");
    assertThat(TreeUtils.nameFromExpression(lastExpression("self.my_var"))).isNullOrEmpty();
    assertThat(TreeUtils.nameFromExpression(lastExpression("my_call()"))).isNullOrEmpty();
    assertThat(TreeUtils.nameFromExpression(lastExpression("a == b"))).isNullOrEmpty();
  }

  @Test
  void test_nameFromQualifiedOrCallExpression() {
    assertThat(TreeUtils.nameFromQualifiedOrCallExpression(lastExpression("my_var"))).contains("my_var");
    assertThat(TreeUtils.nameFromQualifiedOrCallExpression(lastExpression("self.my_var"))).contains("self.my_var");
    assertThat(TreeUtils.nameFromQualifiedOrCallExpression(lastExpression("my_call()"))).contains("my_call");
    assertThat(TreeUtils.nameFromQualifiedOrCallExpression(lastExpression("a == b"))).isNotPresent();
  }

  @Test
  void test_fullyQualifiedNameFromQualifiedExpression() {
    FileInput fileInput = PythonTestUtils.parse(
      "from third_party_lib import (element as alias)",
      "a = alias.attribute");
    QualifiedExpression qualifiedExpression = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.QUALIFIED_EXPR));
    assertThat(TreeUtils.fullyQualifiedNameFromQualifiedExpression(qualifiedExpression)).contains("third_party_lib.element.attribute");

    fileInput = PythonTestUtils.parse(
      "from third_party_lib import (element as alias)",
      "a = alias().attribute");
    qualifiedExpression = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.QUALIFIED_EXPR));
    assertThat(TreeUtils.fullyQualifiedNameFromQualifiedExpression(qualifiedExpression)).contains("third_party_lib.element.attribute");

    fileInput = PythonTestUtils.parse(
      "from third_party_lib import (element as alias)",
      "a = alias.attr");
    qualifiedExpression = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.QUALIFIED_EXPR));
    assertThat(TreeUtils.fullyQualifiedNameFromQualifiedExpression(qualifiedExpression)).contains("third_party_lib.element.attr");
  }

  @Test
  void test_getTreeSeparatorOrLastToken() {
    FileInput fileInput = PythonTestUtils.parse("a = 1");
    Token lastToken = TreeUtils.getTreeSeparatorOrLastToken(fileInput.statements().statements().get(0));
    assertThat(lastToken.type().getName()).isEqualTo("NUMBER");

    fileInput = PythonTestUtils.parse("a = 1;");
    lastToken = TreeUtils.getTreeSeparatorOrLastToken(fileInput.statements().statements().get(0));
    assertThat(lastToken.type().getName()).isEqualTo("SEMICOLON");

    fileInput = PythonTestUtils.parse("a = 1", "");
    lastToken = TreeUtils.getTreeSeparatorOrLastToken(fileInput.statements().statements().get(0));
    assertThat(lastToken.type().getName()).isEqualTo("NEWLINE");
  }

  @Test
  void test_isFunctionWithGivenDecoratorFQN() {
    FileInput fileInput = PythonTestUtils.parse(
      """
        import some_module
        @some_module.some_decorator
        def foo():
          ...
        """);
    assertThat(TreeUtils.isFunctionWithGivenDecoratorFQN(fileInput, "some_module.some_decorator")).isFalse();
    FunctionDef functionDef = ((FunctionDef) fileInput.statements().statements().get(1));
    assertThat(TreeUtils.isFunctionWithGivenDecoratorFQN(functionDef, "some_module.some_decorator")).isTrue();
    assertThat(TreeUtils.isFunctionWithGivenDecoratorFQN(functionDef, "some_module.unknown")).isFalse();
  }

  @Test
  void test_groupAssignmentByParentStatementList() {
    FileInput fileInput = PythonTestUtils.parse("def foo(a):\n" +
      "    b = a\n" +
      "    if a > 10:\n" +
      "        b = 10\n" +
      "    c = 3 ");

    var fooDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF) && ((FunctionDef) t).name().name().equals("foo"));
    var assignments = PythonTestUtils.getAllDescendant(fooDef, t -> t.is(Kind.ASSIGNMENT_STMT));
    var grouped = assignments.stream()
      .collect(TreeUtils.groupAssignmentByParentStatementList());

    assertThat(grouped).hasSize(2);
  }

  @Test
  void test_getTreeByPositionComparator() {
    FileInput fileInput = PythonTestUtils.parse("def foo(a):\n" +
      "    b = a\n" +
      "    if a > 10:\n" +
      "        b = 10\n");

    var fooDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF) && ((FunctionDef) t).name().name().equals("foo"));
    var assignments = PythonTestUtils.getAllDescendant(fooDef, t -> t.is(Kind.ASSIGNMENT_STMT));
    var comparator = TreeUtils.getTreeByPositionComparator();

    int comparsionResult = comparator.compare(assignments.get(1), assignments.get(0));
    assertThat(comparsionResult).isPositive();
  }

  @Test
  void test_toOptionalInstanceOf() {
    var fileInput = PythonTestUtils.parse(
      "class A:",
      "    x = True",
      "    def foo(self):",
      "        def foo2(x, y): return x + y",
      "        return foo2(1, 1)",
      "    class B:",
      "        def bar(self): pass");
    Tree tree = PythonTestUtils.getFirstChild(fileInput, t -> t.is(Kind.CLASSDEF));

    boolean classPresent = TreeUtils.toOptionalInstanceOf(ClassDef.class, tree)
      .isPresent();

    assertThat(classPresent).isTrue();

    boolean funcDefPresent = TreeUtils.toOptionalInstanceOf(FunctionDef.class, tree)
      .isPresent();

    assertThat(funcDefPresent).isFalse();
  }

  @Test
  void test_toOptionalInstanceOfMapper() {
    var fileInput = PythonTestUtils.parse(
      "class A:",
      "    x = True",
      "    def foo(self):",
      "        def foo2(x, y): return x + y",
      "        return foo2(1, 1)",
      "    class B:",
      "        def bar(self): pass");
    Tree tree = PythonTestUtils.getFirstChild(fileInput, t -> t.is(Kind.CLASSDEF));

    boolean classPresent = Optional.of(tree)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(ClassDef.class))
      .isPresent();

    assertThat(classPresent).isTrue();

    boolean funcDefPresent = Optional.of(tree)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(FunctionDef.class))
      .isPresent();

    assertThat(funcDefPresent).isFalse();
  }

  @Test
  void test_toInstanceOfMapper() {
    var fileInput = PythonTestUtils.parse(
      "class A:",
      "    x = True",
      "    def foo(self):",
      "        def foo2(x, y): return x + y",
      "        return foo2(1, 1)",
      "    class B:",
      "        def bar(self): pass");
    Tree tree = PythonTestUtils.getFirstChild(fileInput, t -> t.is(Kind.CLASSDEF));

    boolean classPresent = Optional.of(tree)
      .map(TreeUtils.toInstanceOfMapper(ClassDef.class))
      .isPresent();

    assertThat(classPresent).isTrue();

    boolean funcDefPresent = Optional.of(tree)
      .map(TreeUtils.toInstanceOfMapper(FunctionDef.class))
      .isPresent();

    assertThat(funcDefPresent).isFalse();
  }

  @Test
  void test_toStreamInstanceOfMapper() {
    var fileInput = PythonTestUtils.parse(
      "class A:",
      "    x = True",
      "    def foo(self):",
      "        def foo2(x, y): return x + y",
      "        return foo2(1, 1)",
      "    class B:",
      "        def bar(self): pass");
    Tree tree = PythonTestUtils.getFirstChild(fileInput, t -> t.is(Kind.CLASSDEF));

    boolean classPresent = Stream.of(tree)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(ClassDef.class))
      .count() > 0;

    assertThat(classPresent).isTrue();

    boolean funcDefPresent = Stream.of(tree)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(FunctionDef.class))
      .count() > 0;

    assertThat(funcDefPresent).isFalse();
  }

  @Test
  void test_findIndentationSize() {
    var fileInput = PythonTestUtils.parse("def foo():\n" +
      "    if a < 3: pass\n");

    var passDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.PASS_STMT));

    var indent = TreeUtils.findIndentationSize(passDef);
    assertThat(indent).isEqualTo(4);

    fileInput = PythonTestUtils.parse("class A():\n" +
      "    def foo(self):\n" +
      "      if a < 3: pass\n");

    passDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.PASS_STMT));

    indent = TreeUtils.findIndentationSize(passDef);
    assertThat(indent).isEqualTo(2);

  }

  @Test
  void test_findIndentationSizeDownTree() {
    var fileInput = PythonTestUtils.parse("if a < 3: pass\n" +
      "\n" +
      "def foo(a):\n" +
      "  print(a)");

    var passDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.PASS_STMT));

    var indent = TreeUtils.findIndentationSize(passDef);
    assertThat(indent).isEqualTo(2);

    fileInput = PythonTestUtils.parse("if a < 3: pass\n" +
      "\n" +
      "class A():\n" +
      "    def foo(self, a):\n" +
      "      print(a)");

    passDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.PASS_STMT));

    indent = TreeUtils.findIndentationSize(passDef);
    assertThat(indent).isEqualTo(4);
  }

  @Test
  void test_findIndentationSizeZero() {
    var fileInput = PythonTestUtils.parse("if a < 3: pass\n" +
      "\n" +
      "def foo(a): pass");

    var passDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.PASS_STMT));

    var indent = TreeUtils.findIndentationSize(passDef);
    assertThat(indent).isZero();
  }

  @Test
  void test_findIndentationSizeEmpty() {
    var fileInput = PythonTestUtils.parse("");
    var indent = TreeUtils.findIndentationSize(fileInput);
    assertThat(indent).isZero();
  }

  @Test
  void test_firstChild() {
    var fileInput = PythonTestUtils.parse(
      "class A:",
      "    x = True",
      "    def foo(self):",
      "        def foo2(x, y): return x + y",
      "        return foo2(1, 1)",
      "    class B:",
      "        def bar(self): pass");
    var functionOpt = TreeUtils.firstChild(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(functionOpt).isPresent();

    var classDefOpt = TreeUtils.firstChild(functionOpt.get(), t -> t.is(Kind.CLASSDEF));
    assertThat(classDefOpt).isNotPresent();
  }

  @Test
  void treeToStringTest() {
    var input = "a = 1\n" +
      "b = 2";
    var fileInput = PythonTestUtils.parse(input);
    var statements = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.STATEMENT_LIST));

    var result = TreeUtils.treeToString(statements, true);
    assertThat(result).isEqualTo(input);

    result = TreeUtils.treeToString(statements, false);
    assertThat(result).isNull();

    input = "a = 1";
    fileInput = PythonTestUtils.parse(input);
    statements = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.STATEMENT_LIST));
    result = TreeUtils.treeToString(statements, false);
    assertThat(result).isEqualTo(input);
  }

  @Test
  void dottedNameToPart() {
    var input = "import a.b";
    var fileInput = PythonTestUtils.parse(input);
    var statements = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.DOTTED_NAME));
    var result = TreeUtils.dottedNameToPartFqn((DottedName) statements);
    assertThat(result).hasSameElementsAs(List.of("a", "b"));
    input = "import mod";
    fileInput = PythonTestUtils.parse(input);
    statements = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.DOTTED_NAME));
    result = TreeUtils.dottedNameToPartFqn((DottedName) statements);
    assertThat(result).hasSameElementsAs(List.of("mod"));
    input = "from a import *";
    fileInput = PythonTestUtils.parse(input);
    statements = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.DOTTED_NAME));
    result = TreeUtils.dottedNameToPartFqn((DottedName) statements);
    assertThat(result).hasSameElementsAs(List.of("a"));
  }

  private static boolean isOuterFunction(Tree tree) {
    return tree.is(Kind.FUNCDEF) && ((FunctionDef) tree).name().name().equals("outer");
  }

  private FileInput parse(String content) {
    PythonParser parser = PythonParser.create();
    AstNode astNode = parser.parse(content);
    return new PythonTreeMaker().fileInput(astNode);
  }
}
