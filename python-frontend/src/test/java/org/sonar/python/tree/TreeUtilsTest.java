/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.PassStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.parser.PythonParser;

import static org.assertj.core.api.Assertions.assertThat;

public class TreeUtilsTest {

  @Test
  public void first_ancestor_of_kind() {
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
  public void first_ancestor() {
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
  public void tokens() {
    // simple statement parsed so that we easily get all tokens from children or first token.
    FileInput parsed = parse("if foo:\n  pass");
    IfStatement ifStmt = (IfStatement) parsed.statements().statements().get(0);
    List<Token> collect = new ArrayList<>(ifStmt.children().stream().map(t -> t.is(Kind.TOKEN) ? (Token) t : t.firstToken()).collect(Collectors.toList()));
    collect.add(parsed.lastToken());
    assertThat(TreeUtils.tokens(parsed)).containsExactly(collect.toArray(new Token[0]));

    assertThat(TreeUtils.tokens(parsed.lastToken())).containsExactly(parsed.lastToken());

  }

  @Test
  public void non_whitespace_tokens() {
    FileInput parsed = parse("if foo:\n  pass");
    IfStatement ifStmt = (IfStatement) parsed.statements().statements().get(0);
    List<Token> nonWhitespaceTokens = TreeUtils.nonWhitespaceTokens(ifStmt);
    nonWhitespaceTokens.forEach(t -> assertThat(t.type()).isNotIn(PythonTokenType.NEWLINE, PythonTokenType.INDENT, PythonTokenType.DEDENT));
    assertThat(nonWhitespaceTokens).hasSize(4);
    assertThat(nonWhitespaceTokens.stream().map(Token::value)).containsExactly("if", "foo", ":", "pass");
  }

  @Test
  public void hasDescendants() {
    FileInput fileInput = parse("class A:\n  def foo(): pass");
    assertThat(TreeUtils.hasDescendant(fileInput, t -> t.is(Kind.PASS_STMT))).isTrue();
    assertThat(TreeUtils.hasDescendant(fileInput, t -> (t.is(Kind.NAME) && ((Name) t).name().equals("foo")))).isTrue();
    assertThat(TreeUtils.hasDescendant(fileInput, t -> (t.is(Kind.NAME) && ((Name) t).name().equals("bar")))).isFalse();
    assertThat(TreeUtils.hasDescendant(fileInput, t -> t.is(Kind.IF_STMT))).isFalse();
  }

  @Test
  public void getClassSymbolFromDef() {
    FileInput fileInput = PythonTestUtils.parse("class A:\n  def foo(): pass");
    ClassDef classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));

    Symbol symbolA = classDef.name().symbol();
    assertThat(TreeUtils.getClassSymbolFromDef(classDef)).isEqualTo(symbolA);
    assertThat(TreeUtils.getClassSymbolFromDef(null)).isNull();

    fileInput = PythonTestUtils.parse(
      "class A:",
      "    pass",
      "A = 42"
    );
    classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));
    assertThat(TreeUtils.getClassSymbolFromDef(classDef)).isNull();
  }

  @Test(expected = IllegalStateException.class)
  public void getClassSymbolFromDef_illegalSymbol() {
    FileInput fileInput = PythonTestUtils.parseWithoutSymbols("class A:\n  def foo(): pass");
    ClassDef classDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.CLASSDEF));

    TreeUtils.getClassSymbolFromDef(classDef);
  }

  @Test
  public void nonTupleParameters() {
    FileInput fileInput = PythonTestUtils.parse("def foo(): pass");
    FunctionDef functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.nonTupleParameters(functionDef)).isEmpty();

    fileInput = PythonTestUtils.parse("def foo(param1, param2): pass");
    functionDef = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Kind.FUNCDEF));
    assertThat(TreeUtils.nonTupleParameters(functionDef)).isEqualTo(functionDef.parameters().nonTuple());
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
