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
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.AwaitExpression;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.BreakStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.ComprehensionIf;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.ContinueStatement;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.EllipsisExpression;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.ExecStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FormatSpecifier;
import org.sonar.plugins.python.api.tree.FormattedExpression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.GlobalStatement;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.ImportStatement;
import org.sonar.plugins.python.api.tree.InExpression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.MatchStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NoneExpression;
import org.sonar.plugins.python.api.tree.NonlocalStatement;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.PassStatement;
import org.sonar.plugins.python.api.tree.PrintStatement;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.ReprExpression;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.SliceItem;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.TupleParameter;
import org.sonar.plugins.python.api.tree.TypeAliasStatement;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.tree.TypeParams;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.parser.RuleTest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.fail;

class PythonTreeMakerTest extends RuleTest {

  private final PythonTreeMaker treeMaker = new PythonTreeMaker();

  @Test
  void file_input() {
    FileInput fileInput = parse("", treeMaker::fileInput);
    assertThat(fileInput.statements()).isNull();
    assertThat(fileInput.docstring()).isNull();

    fileInput = parse("\"\"\"\n" +
      "This is a module docstring\n" +
      "\"\"\"", treeMaker::fileInput);
    assertThat(fileInput.docstring().stringElements()).hasSize(1);
    assertThat(fileInput.docstring().stringElements().get(0).value()).isEqualTo("\"\"\"\n" +
      "This is a module docstring\n" +
      "\"\"\"");
    assertThat(fileInput.children()).hasSize(2);

    fileInput = parse("if x:\n pass", treeMaker::fileInput);
    IfStatement ifStmt = (IfStatement) fileInput.statements().statements().get(0);
    assertThat(ifStmt.body().parent()).isEqualTo(ifStmt);
    assertThat(fileInput.children()).hasSize(2);
    assertThat(((Token) fileInput.children().get(1)).type()).isEqualTo(GenericTokenType.EOF);
  }

  @Test
  void variadic_is_kind() {
    FileInput fileInput = parse("def foo(): pass", treeMaker::fileInput);
    assertThat(fileInput.is(Tree.Kind.FILE_INPUT, Tree.Kind.STATEMENT_LIST)).isTrue();
    FunctionDef functionDef = (FunctionDef) fileInput.statements().statements().get(0);
    assertThat(functionDef.is(Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF)).isTrue();
    assertThat(functionDef.is(Tree.Kind.WHILE_STMT, Tree.Kind.CLASSDEF)).isFalse();
  }

  @Test
  void unexpected_expression_should_throw_an_exception() {
    try {
      parse("", treeMaker::expression);
      fail("unexpected ASTNode type for expression should not succeed to be translated to Strongly typed AST");
    } catch (IllegalStateException iae) {
      assertThat(iae).hasMessage("Expression FILE_INPUT not correctly translated to strongly typed AST");
    }
  }

  @Test
  void verify_expected_statement() {
    Map<String, Class<? extends Tree>> testData = new HashMap<>();
    testData.put("pass", PassStatement.class);
    testData.put("print 'foo'", PrintStatement.class);
    testData.put("exec foo", ExecStatement.class);
    testData.put("assert foo", AssertStatement.class);
    testData.put("del foo", DelStatement.class);
    testData.put("return foo", ReturnStatement.class);
    testData.put("yield foo", YieldStatement.class);
    testData.put("raise foo", RaiseStatement.class);
    testData.put("break", BreakStatement.class);
    testData.put("continue", ContinueStatement.class);
    testData.put("def foo():pass", FunctionDef.class);
    testData.put("import foo", ImportStatement.class);
    testData.put("from foo import f", ImportStatement.class);
    testData.put("class toto:pass", ClassDef.class);
    testData.put("for foo in bar:pass", ForStatement.class);
    testData.put("for foo1, foo2 in bar:pass", ForStatement.class);
    testData.put("for foo in k:=bar:pass", ForStatement.class);
    testData.put("for foo in *a, *b, *c:pass", ForStatement.class);
    testData.put("async for foo in bar: pass", ForStatement.class);
    testData.put("global foo", GlobalStatement.class);
    testData.put("nonlocal foo", NonlocalStatement.class);
    testData.put("while cond: pass", WhileStatement.class);
    testData.put("'foo'", ExpressionStatement.class);
    testData.put("try: this\nexcept Exception: pass", TryStatement.class);
    testData.put("with foo, bar as qix : pass", WithStatement.class);
    testData.put("async with foo, bar as qix : pass", WithStatement.class);
    testData.put("x = y", AssignmentStatement.class);
    testData.put("x += y", CompoundAssignmentStatement.class);
    testData.put("match command:\n  case 42:...\n", MatchStatement.class);
    testData.put("type A[B] = str", TypeAliasStatement.class);

    testData.forEach((c, clazz) -> {
      FileInput pyTree = parse(c, treeMaker::fileInput);
      StatementList statementList = pyTree.statements();
      assertThat(statementList.statements()).hasSize(1);
      Statement stmt = statementList.statements().get(0);
      assertThat(stmt.parent()).isEqualTo(statementList);
      assertThat(stmt).as(c).isInstanceOf(clazz);
    });
  }

  @Test
  void IfStatement() {
    setRootRule(PythonGrammar.IF_STMT);
    IfStatement pyIfStatementTree = parse("if x: pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().value()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(Expression.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.body().statements()).hasSize(1);
    assertThat(pyIfStatementTree.children()).hasSize(4);


    pyIfStatementTree = parse("if x: pass\nelse: pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().value()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(Expression.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    ElseClause elseBranch = pyIfStatementTree.elseBranch();
    assertThat(elseBranch.firstToken().value()).isEqualTo("else");
    assertThat(elseBranch.lastToken().value()).isEqualTo("pass");
    assertThat(elseBranch).isNotNull();
    assertThat(elseBranch.elseKeyword().value()).isEqualTo("else");
    assertThat(elseBranch.body().statements()).hasSize(1);
    assertThat(pyIfStatementTree.children()).hasSize(5);


    pyIfStatementTree = parse("if x: pass\nelif y: pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().value()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(Expression.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.elifBranches()).hasSize(1);
    IfStatement elif = pyIfStatementTree.elifBranches().get(0);
    assertThat(elif.condition()).isInstanceOf(Expression.class);
    assertThat(elif.firstToken().value()).isEqualTo("elif");
    assertThat(elif.lastToken().value()).isEqualTo("pass");
    assertThat(elif.isElif()).isTrue();
    assertThat(elif.elseBranch()).isNull();
    assertThat(elif.elifBranches()).isEmpty();
    assertThat(elif.body().statements()).hasSize(1);
    assertThat(pyIfStatementTree.children()).hasSize(5);

    pyIfStatementTree = parse("if x:\n pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().value()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(Expression.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    assertThat(pyIfStatementTree.body().statements()).hasSize(1);

    pyIfStatementTree = parse("if x:\n pass\n pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.body().statements()).hasSize(2);

  }

  @Test
  void printStatement() {
    setRootRule(PythonGrammar.PRINT_STMT);
    AstNode astNode = p.parse("print 'foo'");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    PrintStatement printStmt = treeMaker.printStatement(statementWithSeparator);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().value()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(1);
    assertThat(printStmt.children()).hasSize(2);

    astNode = p.parse("print 'foo', 'bar'");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    printStmt = treeMaker.printStatement(statementWithSeparator);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().value()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(2);
    assertThat(printStmt.children()).hasSize(3);

    astNode = p.parse("print >> 'foo'");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    printStmt = treeMaker.printStatement(statementWithSeparator);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().value()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(1);
  }

  @Test
  void execStatement() {
    setRootRule(PythonGrammar.EXEC_STMT);
    AstNode astNode = p.parse("exec 'foo'");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ExecStatement execStatement = treeMaker.execStatement(statementWithSeparator);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().value()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNull();
    assertThat(execStatement.localsExpression()).isNull();
    assertThat(execStatement.children()).hasSize(2);

    astNode = p.parse("exec 'foo' in globals");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    execStatement = treeMaker.execStatement(statementWithSeparator);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().value()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNotNull();
    assertThat(execStatement.localsExpression()).isNull();
    assertThat(execStatement.children()).extracting(child -> child.firstToken().value())
      .containsExactly("exec", "'foo'", "in", "globals");

    astNode = p.parse("exec 'foo' in globals, locals");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    execStatement = treeMaker.execStatement(statementWithSeparator);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().value()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNotNull();
    assertThat(execStatement.localsExpression()).isNotNull();
    assertThat(execStatement.children()).extracting(child -> child.firstToken().value())
      .containsExactly("exec", "'foo'", "in", "globals", ",", "locals");

    // TODO: exec stmt should parse exec ('foo', globals, locals); see https://docs.python.org/2/reference/simple_stmts.html#exec
  }

  @Test
  void assertStatement() {
    setRootRule(PythonGrammar.ASSERT_STMT);
    AstNode astNode = p.parse("assert x");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AssertStatement assertStatement = treeMaker.assertStatement(statementWithSeparator);
    assertThat(assertStatement).isNotNull();
    assertThat(assertStatement.assertKeyword().value()).isEqualTo("assert");
    assertThat(assertStatement.condition()).isNotNull();
    assertThat(assertStatement.message()).isNull();
    assertThat(assertStatement.children()).hasSize(2);

    astNode = p.parse("assert x, y");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    assertStatement = treeMaker.assertStatement(statementWithSeparator);
    assertThat(assertStatement).isNotNull();
    assertThat(assertStatement.assertKeyword().value()).isEqualTo("assert");
    assertThat(assertStatement.condition()).isNotNull();
    assertThat(assertStatement.message()).isNotNull();
    assertThat(assertStatement.children()).hasSize(3);
  }

  @Test
  void passStatement() {
    setRootRule(PythonGrammar.PASS_STMT);
    AstNode astNode = p.parse("pass");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    PassStatement passStatement = treeMaker.passStatement(statementWithSeparator);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.passKeyword().value()).isEqualTo("pass");
    assertThat(passStatement.children()).hasSize(1);
  }

  @Test
  void delStatement() {
    setRootRule(PythonGrammar.DEL_STMT);
    AstNode astNode = p.parse("del foo");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    DelStatement passStatement = treeMaker.delStatement(statementWithSeparator);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().value()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);
    assertThat(passStatement.children()).hasSize(2);


    astNode = p.parse("del foo, bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    passStatement = treeMaker.delStatement(statementWithSeparator);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().value()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(2);
    assertThat(passStatement.children()).hasSize(3);

    astNode = p.parse("del *foo");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    passStatement = treeMaker.delStatement(statementWithSeparator);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().value()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);
    assertThat(passStatement.children()).hasSize(2);
  }

  @Test
  void returnStatement() {
    setRootRule(PythonGrammar.RETURN_STMT);
    AstNode astNode = p.parse("return foo");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ReturnStatement returnStatement = treeMaker.returnStatement(statementWithSeparator);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().value()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(1);
    assertThat(returnStatement.commas()).isEmpty();
    assertThat(returnStatement.children()).hasSize(2);

    astNode = p.parse("return foo, bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    returnStatement = treeMaker.returnStatement(statementWithSeparator);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().value()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(2);
    assertThat(returnStatement.commas()).hasSize(1);
    assertThat(returnStatement.children()).hasSize(4);
    assertThat(returnStatement.children().get(2).is(Kind.TOKEN)).isTrue();

    astNode = p.parse("return");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    returnStatement = treeMaker.returnStatement(statementWithSeparator);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().value()).isEqualTo("return");
    assertThat(returnStatement.expressions()).isEmpty();
    assertThat(returnStatement.commas()).isEmpty();
    assertThat(returnStatement.children()).hasSize(1);

    astNode = p.parse("return []");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    returnStatement = treeMaker.returnStatement(statementWithSeparator);
    ListLiteral listLiteral = (ListLiteral) returnStatement.expressions().get(0);
    assertThat(listLiteral.leftBracket()).isSameAs(listLiteral.children().get(0));
    assertThat(listLiteral.rightBracket()).isSameAs(listLiteral.children().get(2));
    ExpressionList emptyExpressionList = listLiteral.elements();
    assertThat(emptyExpressionList.children()).isEmpty();
    assertThat(emptyExpressionList.firstToken()).isNull();
    assertThat(emptyExpressionList.lastToken()).isNull();

    astNode = p.parse("return foo(), *bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    returnStatement = treeMaker.returnStatement(statementWithSeparator);
    assertThat(returnStatement.expressions()).extracting(Tree::getKind).containsExactly(Kind.CALL_EXPR, Kind.UNPACKING_EXPR);
  }

  @Test
  void yieldStatement() {
    setRootRule(PythonGrammar.YIELD_STMT);
    AstNode astNode = p.parse("yield foo");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    YieldStatement yieldStatement = treeMaker.yieldStatement(statementWithSeparator);
    assertThat(yieldStatement).isNotNull();
    assertThat(yieldStatement.children()).hasSize(1);
    YieldExpression yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(YieldExpression.class);
    assertThat(yieldExpression.expressions()).hasSize(1);
    assertThat(yieldExpression.children()).hasSize(2);

    astNode = p.parse("yield foo, bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    yieldStatement = treeMaker.yieldStatement(statementWithSeparator);
    assertThat(yieldStatement).isNotNull();
    assertThat(yieldStatement.children()).hasSize(1);
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(YieldExpression.class);
    assertThat(yieldExpression.yieldKeyword().value()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword()).isNull();
    assertThat(yieldExpression.expressions()).hasSize(2);
    assertThat(yieldExpression.children()).hasSize(3);

    astNode = p.parse("yield from foo");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    yieldStatement = treeMaker.yieldStatement(statementWithSeparator);
    assertThat(yieldStatement).isNotNull();
    assertThat(yieldStatement.children()).hasSize(1);
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(YieldExpression.class);
    assertThat(yieldExpression.yieldKeyword().value()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword().value()).isEqualTo("from");
    assertThat(yieldExpression.expressions()).hasSize(1);
    assertThat(yieldExpression.children()).hasSize(3);

    astNode = p.parse("yield");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    yieldStatement = treeMaker.yieldStatement(statementWithSeparator);
    assertThat(yieldStatement.children()).hasSize(1);
    assertThat(yieldStatement).isNotNull();

    astNode = p.parse("yield foo(), *bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    yieldStatement = treeMaker.yieldStatement(statementWithSeparator);
    assertThat(yieldStatement.yieldExpression().expressions()).extracting(Tree::getKind).containsExactly(Kind.CALL_EXPR, Kind.UNPACKING_EXPR);
  }

  @Test
  void raiseStatement() {
    setRootRule(PythonGrammar.RAISE_STMT);
    AstNode astNode = p.parse("raise foo");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    RaiseStatement raiseStatement = treeMaker.raiseStatement(statementWithSeparator);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().value()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).hasSize(1);
    assertThat(raiseStatement.children()).hasSize(2);

    astNode = p.parse("raise foo, bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    raiseStatement = treeMaker.raiseStatement(statementWithSeparator);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().value()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).hasSize(2);
    assertThat(raiseStatement.children()).hasSize(3);

    astNode = p.parse("raise foo from bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    raiseStatement = treeMaker.raiseStatement(statementWithSeparator);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().value()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword().value()).isEqualTo("from");
    assertThat(raiseStatement.fromExpression()).isNotNull();
    assertThat(raiseStatement.expressions()).hasSize(1);
    assertThat(raiseStatement.children()).hasSize(4);

    astNode = p.parse("raise");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    raiseStatement = treeMaker.raiseStatement(statementWithSeparator);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().value()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).isEmpty();
    assertThat(raiseStatement.children()).hasSize(1);
  }

  @Test
  void breakStatement() {
    setRootRule(PythonGrammar.BREAK_STMT);
    AstNode astNode = p.parse("break");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    BreakStatement breakStatement = treeMaker.breakStatement(statementWithSeparator);
    assertThat(breakStatement).isNotNull();
    assertThat(breakStatement.breakKeyword().value()).isEqualTo("break");
    assertThat(breakStatement.children()).hasSize(1);
  }

  @Test
  void continueStatement() {
    setRootRule(PythonGrammar.CONTINUE_STMT);
    AstNode astNode = p.parse("continue");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ContinueStatement continueStatement = treeMaker.continueStatement(statementWithSeparator);
    assertThat(continueStatement).isNotNull();
    assertThat(continueStatement.continueKeyword().value()).isEqualTo("continue");
    assertThat(continueStatement.children()).hasSize(1);
  }

  @Test
  void importStatement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    AstNode astNode = p.parse("import foo");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ImportName importStatement = (ImportName) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement.firstToken().value()).isEqualTo("import");
    assertThat(importStatement.lastToken().value()).isEqualTo("foo");
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().value()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    AliasedName importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("import foo as f");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportName) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.firstToken().value()).isEqualTo("import");
    assertThat(importStatement.lastToken().value()).isEqualTo("f");
    assertThat(importStatement.importKeyword().value()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.firstToken().value()).isEqualTo("foo");
    assertThat(importedName1.lastToken().value()).isEqualTo("f");
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importedName1.asKeyword().value()).isEqualTo("as");
    assertThat(importedName1.alias().name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("import foo.bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportName) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().value()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(2);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importedName1.dottedName().names().get(1).name()).isEqualTo("bar");
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("import foo, bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportName) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().value()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(2);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    AliasedName importedName2 = importStatement.modules().get(1);
    assertThat(importedName2.dottedName().names()).hasSize(1);
    assertThat(importedName2.dottedName().names().get(0).name()).isEqualTo("bar");
    assertThat(importStatement.children()).hasSize(3);
  }

  @Test
  void importFromStatement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    AstNode astNode = p.parse("from foo import f");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ImportFrom importStatement = (ImportFrom) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.firstToken().value()).isEqualTo("from");
    assertThat(importStatement.lastToken().value()).isEqualTo("f");
    assertThat(importStatement.importKeyword().value()).isEqualTo("import");
    assertThat(importStatement.dottedPrefixForModule()).isEmpty();
    assertThat(importStatement.fromKeyword().value()).isEqualTo("from");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.isWildcardImport()).isFalse();
    assertThat(importStatement.wildcard()).isNull();
    assertThat(importStatement.importedNames()).hasSize(1);
    AliasedName aliasedNameTree = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree.asKeyword()).isNull();
    assertThat(aliasedNameTree.alias()).isNull();
    assertThat(aliasedNameTree.dottedName().names().get(0).name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(4);
    assertThat(importStatement.children()).containsExactly(importStatement.fromKeyword(), importStatement.module(), importStatement.importKeyword(), aliasedNameTree);


    astNode = p.parse("from .foo import f");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportFrom) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).value()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");
    aliasedNameTree = importStatement.importedNames().get(0);
    assertThat(importStatement.children()).hasSize(5);
    assertThat(importStatement.children()).containsExactly(importStatement.fromKeyword(), importStatement.dottedPrefixForModule().get(0),
      importStatement.module(), importStatement.importKeyword(), aliasedNameTree);

    astNode = p.parse("from ..foo import f");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportFrom) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(2);
    assertThat(importStatement.dottedPrefixForModule().get(0).value()).isEqualTo(".");
    assertThat(importStatement.dottedPrefixForModule().get(1).value()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.children()).hasSize(6);

    astNode = p.parse("from . import f");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportFrom) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).value()).isEqualTo(".");
    assertThat(importStatement.module()).isNull();
    assertThat(importStatement.children()).hasSize(4);

    astNode = p.parse("from foo import f as g");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportFrom) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement.importedNames()).hasSize(1);
    aliasedNameTree = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree.asKeyword().value()).isEqualTo("as");
    assertThat(aliasedNameTree.alias().name()).isEqualTo("g");
    assertThat(aliasedNameTree.dottedName().names().get(0).name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(4);

    astNode = p.parse("from foo import f as g, h");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportFrom) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement.importedNames()).hasSize(2);
    AliasedName aliasedNameTree1 = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree1.asKeyword().value()).isEqualTo("as");
    assertThat(aliasedNameTree1.alias().name()).isEqualTo("g");
    assertThat(aliasedNameTree1.dottedName().names().get(0).name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(5);
    List<Tree> aliasedNameChildren = aliasedNameTree1.children();
    assertThat(aliasedNameChildren).hasSize(3);
    assertThat(aliasedNameChildren).containsExactly(aliasedNameTree1.dottedName(), aliasedNameTree1.asKeyword(), aliasedNameTree1.alias());

    AliasedName aliasedNameTree2 = importStatement.importedNames().get(1);
    assertThat(aliasedNameTree2.asKeyword()).isNull();
    assertThat(aliasedNameTree2.alias()).isNull();
    assertThat(aliasedNameTree2.dottedName().names().get(0).name()).isEqualTo("h");

    astNode = p.parse("from foo import *");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportFrom) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement.importedNames()).isEmpty();
    assertThat(importStatement.isWildcardImport()).isTrue();
    assertThat(importStatement.wildcard().value()).isEqualTo("*");
    assertThat(importStatement.children()).hasSize(4);
    assertThat(importStatement.children()).containsExactly(importStatement.fromKeyword(), importStatement.module(), importStatement.importKeyword(), importStatement.wildcard());
  }

  @Test
  void globalStatement() {
    setRootRule(PythonGrammar.GLOBAL_STMT);
    AstNode astNode = p.parse("global foo");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    GlobalStatement globalStatement = treeMaker.globalStatement(statementWithSeparator);
    assertThat(globalStatement.globalKeyword().value()).isEqualTo("global");
    assertThat(globalStatement.variables()).hasSize(1);
    assertThat(globalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(globalStatement.children()).hasSize(2);

    astNode = p.parse("global foo, bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    globalStatement = treeMaker.globalStatement(statementWithSeparator);
    assertThat(globalStatement.globalKeyword().value()).isEqualTo("global");
    assertThat(globalStatement.variables()).hasSize(2);
    assertThat(globalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(globalStatement.variables().get(1).name()).isEqualTo("bar");
    assertThat(globalStatement.children()).hasSize(3);
  }

  @Test
  void nonlocalStatement() {
    setRootRule(PythonGrammar.NONLOCAL_STMT);
    AstNode astNode = p.parse("nonlocal foo");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    NonlocalStatement nonlocalStatement = treeMaker.nonlocalStatement(statementWithSeparator);
    assertThat(nonlocalStatement.nonlocalKeyword().value()).isEqualTo("nonlocal");
    assertThat(nonlocalStatement.variables()).hasSize(1);
    assertThat(nonlocalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(nonlocalStatement.children()).hasSize(2);

    astNode = p.parse("nonlocal foo, bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    nonlocalStatement = treeMaker.nonlocalStatement(statementWithSeparator);
    assertThat(nonlocalStatement.nonlocalKeyword().value()).isEqualTo("nonlocal");
    assertThat(nonlocalStatement.variables()).hasSize(2);
    assertThat(nonlocalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(nonlocalStatement.variables().get(1).name()).isEqualTo("bar");
    assertThat(nonlocalStatement.children()).hasSize(3);
  }

  @Test
  void funcdef_statement_type_params() {
    setRootRule(PythonGrammar.FUNCDEF);
    var functionDef = parse("def overly_generic[\n" +
      "   SimpleTypeVar,\n" +
      "   TypeVarWithBound: int,\n" +
      "   TypeVarWithConstraints: (str, bytes),\n" +
      "   *SimpleTypeVarTuple,\n" +
      "   **SimpleParamSpec\n" +
      "](\n" +
      "   a: SimpleTypeVar,\n" +
      "   b: TypeVarWithBound,\n" +
      "   c: Callable[SimpleParamSpec, TypeVarWithConstraints],\n" +
      "   *d: SimpleTypeVarTuple,\n" +
      "   e: *SimpleTypeVar\n" +
      "): pass", treeMaker::funcDefStatement);
    assertThat(functionDef.name()).isNotNull();
    var typeParams = functionDef.typeParams();
    validateTypeParams(typeParams, functionDef);
  }

  private static void validateTypeParams(@Nullable TypeParams typeParams, Tree parent) {
    assertThat(typeParams).isNotNull();
    assertThat(typeParams.getKind()).isEqualTo(Kind.TYPE_PARAMS);
    assertThat(typeParams.leftBracket()).isNotNull();
    assertThat(typeParams.rightBracket()).isNotNull();
    assertThat(typeParams.children()).hasSize(11);
    var typeParamsList = typeParams.typeParamsList();
    assertThat(typeParamsList).isNotNull().hasSize(5).allMatch(p -> p.is(Kind.TYPE_PARAM));

    var simpleTypeVar = typeParamsList.get(0);
    assertThat(simpleTypeVar.name().name()).isEqualTo("SimpleTypeVar");
    assertThat(simpleTypeVar.starToken()).isNull();
    assertThat(simpleTypeVar.typeAnnotation()).isNull();

    var typeWithBound = typeParamsList.get(1);
    assertThat(typeWithBound.name().name()).isEqualTo("TypeVarWithBound");
    assertThat(typeWithBound.starToken()).isNull();
    var typeAnnotation = typeWithBound.typeAnnotation();
    assertThat(typeAnnotation).isNotNull();
    assertThat(typeAnnotation.getKind()).isNotNull().isEqualTo(Kind.TYPE_PARAM_TYPE_ANNOTATION);
    assertThat(typeAnnotation.expression())
      .isNotNull()
      .matches(e -> e.is(Kind.NAME))
      .extracting(Name.class::cast)
      .extracting(Name::name)
      .isEqualTo("int");

    var simpleTypeVarTuple = typeParamsList.get(3);
    assertThat(simpleTypeVarTuple.name().name()).isEqualTo("SimpleTypeVarTuple");
    assertThat(simpleTypeVarTuple.starToken()).isNotNull();
    assertThat(simpleTypeVarTuple.typeAnnotation()).isNull();
  }

  @Test
  void funcdef_statement() {
    setRootRule(PythonGrammar.FUNCDEF);
    AstNode astNode = p.parse("def func(): pass");
    FunctionDef functionDef = treeMaker.funcDefStatement(astNode);
    assertThat(functionDef.name()).isNotNull();
    assertThat(functionDef.name().name()).isEqualTo("func");
    assertThat(functionDef.body().statements()).hasSize(1);
    assertThat(functionDef.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(functionDef.children()).hasSize(6);
    assertThat(functionDef.parameters()).isNull();
    assertThat(functionDef.isMethodDefinition()).isFalse();
    assertThat(functionDef.docstring()).isNull();
    assertThat(functionDef.decorators()).isEmpty();
    assertThat(functionDef.asyncKeyword()).isNull();
    assertThat(functionDef.returnTypeAnnotation()).isNull();
    assertThat(functionDef.colon().value()).isEqualTo(":");
    assertThat(functionDef.defKeyword().value()).isEqualTo("def");
    assertThat(functionDef.leftPar().value()).isEqualTo("(");
    assertThat(functionDef.rightPar().value()).isEqualTo(")");

    functionDef = parse("def func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDef.parameters().all()).hasSize(1);

    functionDef = parse("async def func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDef.asyncKeyword().value()).isEqualTo("async");

    functionDef = parse("def func(x) -> string : pass", treeMaker::funcDefStatement);
    TypeAnnotation returnType = functionDef.returnTypeAnnotation();
    assertThat(returnType.getKind()).isEqualTo(Tree.Kind.RETURN_TYPE_ANNOTATION);
    assertThat(returnType.expression().getKind()).isEqualTo(Tree.Kind.NAME);

    functionDef = parse("@foo\ndef func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDef.decorators()).hasSize(1);
    Decorator decoratorTree = functionDef.decorators().get(0);
    assertThat(decoratorTree.getKind()).isEqualTo(Tree.Kind.DECORATOR);
    assertThat(decoratorTree.atToken().value()).isEqualTo("@");
    assertThat(decoratorTree.arguments()).isNull();

    functionDef = parse("@foo()\n@bar(1)\ndef func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDef.decorators()).hasSize(2);
    Decorator decoratorTree1 = functionDef.decorators().get(0);
    assertThat(decoratorTree1.arguments()).isNull();
    Decorator decoratorTree2 = functionDef.decorators().get(1);
    assertThat(decoratorTree2.arguments().arguments()).extracting(arg -> ((RegularArgument) arg).expression().getKind()).containsExactly(Tree.Kind.NUMERIC_LITERAL);

    functionDef = parse("def func(x, y): pass", treeMaker::funcDefStatement);
    assertThat(functionDef.parameters().all()).hasSize(2);

    functionDef = parse("def func(x = 'foo', y): pass", treeMaker::funcDefStatement);
    assertThat(functionDef.parameters().all()).isEqualTo(functionDef.parameters().nonTuple());
    List<Parameter> parameters = functionDef.parameters().nonTuple();
    assertThat(parameters).hasSize(2);
    Parameter parameter1 = parameters.get(0);
    assertThat(parameter1.name().name()).isEqualTo("x");
    assertThat(parameter1.equalToken().value()).isEqualTo("=");
    assertThat(parameter1.defaultValue().is(Tree.Kind.STRING_LITERAL)).isTrue();
    assertThat(parameter1.children()).doesNotContainNull();
    Parameter parameter2 = parameters.get(1);
    assertThat(parameter2.equalToken()).isNull();
    assertThat(parameter2.defaultValue()).isNull();
    assertThat(parameter2.children()).doesNotContainNull();

    functionDef = parse("def func(p1, *p2, **p3): pass", treeMaker::funcDefStatement);
    parameters = functionDef.parameters().nonTuple();
    assertThat(parameters).extracting(p -> p.name().name()).containsExactly("p1", "p2", "p3");
    assertThat(parameters).extracting(p -> p.starToken() == null ? null : p.starToken().value()).containsExactly(null, "*", "**");
    Parameter parameter = parameters.get(2);
    assertThat(parameter.children()).containsExactly(parameter.starToken(), parameter.name());
    assertThat(parameter.firstToken()).isSameAs(parameter.starToken());
    assertThat(parameter.lastToken()).isSameAs(parameter.name().lastToken());

    functionDef = parse("def func(x : int, y): pass", treeMaker::funcDefStatement);
    parameters = functionDef.parameters().nonTuple();
    assertThat(parameters).hasSize(2);
    TypeAnnotation annotation = parameters.get(0).typeAnnotation();
    assertThat(annotation.getKind()).isEqualTo(Tree.Kind.PARAMETER_TYPE_ANNOTATION);
    assertThat(((Name) annotation.expression()).name()).isEqualTo("int");
    assertThat(annotation.children()).hasSize(2);
    assertThat(parameters.get(1).typeAnnotation()).isNull();

    functionDef = parse("def func(a, ((b, c), d)): pass", treeMaker::funcDefStatement);
    assertThat(functionDef.parameters().all()).hasSize(2);
    assertThat(functionDef.parameters().all()).extracting(Tree::getKind).containsExactly(Tree.Kind.PARAMETER, Tree.Kind.TUPLE_PARAMETER);
    TupleParameter tupleParam = (TupleParameter) functionDef.parameters().all().get(1);
    assertThat(tupleParam.openingParenthesis().value()).isEqualTo("(");
    assertThat(tupleParam.closingParenthesis().value()).isEqualTo(")");
    assertThat(tupleParam.parameters()).extracting(Tree::getKind).containsExactly(Tree.Kind.TUPLE_PARAMETER, Tree.Kind.PARAMETER);
    assertThat(tupleParam.commas()).extracting(Token::value).containsExactly(",");
    assertThat(tupleParam.children()).hasSize(5);

    functionDef = parse("def func(x : int, y):\n  \"\"\"\n" +
      "This is a function docstring\n" +
      "\"\"\"\n  pass", treeMaker::funcDefStatement);
    assertThat(functionDef.docstring().stringElements().get(0).value()).isEqualTo("\"\"\"\n" +
      "This is a function docstring\n" +
      "\"\"\"");
    assertThat(functionDef.children()).hasSize(10);

    functionDef = parse("def __call__(self, *, manager):\n  pass", treeMaker::funcDefStatement);
    assertThat(functionDef.parameters().all()).hasSize(3);
    functionDef = parse("def __call__(*):\n  pass", treeMaker::funcDefStatement);
    assertThat(functionDef.parameters().all()).extracting(Tree::getKind).containsExactly(Kind.PARAMETER);

    functionDef = parse("def f(a, /): pass", treeMaker::funcDefStatement);
    assertThat(functionDef.parameters().all()).extracting(Tree::getKind).containsExactly(Kind.PARAMETER, Kind.PARAMETER);
    assertThat(((Parameter) functionDef.parameters().all().get(1)).starToken().value()).isEqualTo("/");

    assertThat(funcDef("def func(): ...").parameters()).isNull();
    assertThat(funcDef("def func(a): ...").parameters().all()).hasSize(1);
    assertThat(funcDef("def func(a, b): ...").parameters().all()).hasSize(2);
    assertThat(funcDef("def func(a, *args): ...").parameters().all()).hasSize(2);
    assertThat(funcDef("def func(a, **kwargs): ...").parameters().all()).hasSize(2);
    assertThat(funcDef("def func(a, *args, **kwargs): ...").parameters().all()).hasSize(3);
    assertThat(funcDef("def func(*args): ...").parameters().all()).hasSize(1);
    assertThat(funcDef("def func(**kwargs): ...").parameters().all()).hasSize(1);
    assertThat(funcDef("def func(*args, **kwargs): ...").parameters().all()).hasSize(2);
    assertThat(funcDef("def func(*args, a, **kwargs): ...").parameters().all()).hasSize(3);
    assertThat(funcDef("def func(*): ...").parameters().all()).hasSize(1);
    assertThat(funcDef("def func(*, a): ...").parameters().all()).hasSize(2);
    assertThat(funcDef("def func(a, b, *, c): ...").parameters().all()).hasSize(4);
    assertThat(funcDef("def func(a, b, /): ...").parameters().all()).hasSize(3);
    assertThat(funcDef("def func(a, b, /, c): ...").parameters().all()).hasSize(4);
    assertThat(funcDef("def func(a, b, /, c, *args): ...").parameters().all()).hasSize(5);
    assertThat(funcDef("def func(a, b, /, c, **kwargs): ...").parameters().all()).hasSize(5);
    assertThat(funcDef("def func(a, b, /, *args): ...").parameters().all()).hasSize(4);
    assertThat(funcDef("def func(a, b, /, **kwargs): ...").parameters().all()).hasSize(4);
  }

  @Test
  void decorators() {
    FunctionDef functionDef = funcDef("@foo()\n@bar(1)\ndef func(x): pass");
    List<Decorator> decorators = functionDef.decorators();
    assertThat(decorators).hasSize(2);
    Decorator decorator = decorators.get(0);
    assertThat(TreeUtils.decoratorNameFromExpression(decorator.expression())).isEqualTo("foo");
    assertThat(TreeUtils.fullyQualifiedNameFromExpression(decorator.expression())).contains("foo");
    assertThat(decorator.expression().is(Kind.CALL_EXPR)).isTrue();

    functionDef = funcDef("@foo.bar(1)\ndef func(x): pass");
    decorators = functionDef.decorators();
    assertThat(decorators).hasSize(1);
    decorator = decorators.get(0);
    assertThat(TreeUtils.decoratorNameFromExpression(decorator.expression())).isEqualTo("foo.bar");
    assertThat(TreeUtils.fullyQualifiedNameFromExpression(decorator.expression())).contains("foo.bar");
    assertThat(decorator.expression().is(Kind.CALL_EXPR)).isTrue();

    functionDef = funcDef("@buttons[\"hello\"].clicked.connect\ndef func(x): pass");
    decorators = functionDef.decorators();
    assertThat(decorators).hasSize(1);
    decorator = decorators.get(0);
    assertThat(TreeUtils.decoratorNameFromExpression(decorator.expression())).isNull();
    assertThat(TreeUtils.fullyQualifiedNameFromExpression(decorator.expression())).isNotPresent();
    assertThat(decorator.expression().is(Kind.QUALIFIED_EXPR)).isTrue();

    functionDef = funcDef("@hello() or bye()\ndef func(x): pass");
    decorators = functionDef.decorators();
    assertThat(decorators).hasSize(1);
    decorator = decorators.get(0);
    assertThat(decorator.expression().getKind()).isEqualTo(Kind.OR);
    BinaryExpression binaryExpression = (BinaryExpression) decorator.expression();
    assertThat(binaryExpression.leftOperand().getKind()).isEqualTo(Kind.CALL_EXPR);
    assertThat(binaryExpression.rightOperand().getKind()).isEqualTo(Kind.CALL_EXPR);

    functionDef = funcDef("@b:=some_decorator\ndef func(x): pass");
    decorators = functionDef.decorators();
    assertThat(decorators).hasSize(1);
    decorator = decorators.get(0);
    assertThat(decorator.expression().getKind()).isEqualTo(Kind.ASSIGNMENT_EXPRESSION);
  }

  private FunctionDef funcDef(String code) {
    setRootRule(PythonGrammar.FUNCDEF);
    return parse(code, treeMaker::funcDefStatement);
  }

  @Test
  void classdef_statement() {
    setRootRule(PythonGrammar.CLASSDEF);
    AstNode astNode = p.parse("class clazz(Parent): pass");
    ClassDef classDef = treeMaker.classDefStatement(astNode);
    assertThat(classDef.name()).isNotNull();
    assertThat(classDef.docstring()).isNull();
    assertThat(classDef.classKeyword().value()).isEqualTo("class");
    assertThat(classDef.leftPar().value()).isEqualTo("(");
    assertThat(classDef.rightPar().value()).isEqualTo(")");
    assertThat(classDef.colon().value()).isEqualTo(":");
    assertThat(classDef.name().name()).isEqualTo("clazz");
    assertThat(classDef.body().statements()).hasSize(1);
    assertThat(classDef.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(classDef.args().is(Tree.Kind.ARG_LIST)).isTrue();
    assertThat(classDef.args().children()).hasSize(1);
    assertThat(classDef.args().arguments().get(0).is(Tree.Kind.REGULAR_ARGUMENT)).isTrue();
    assertThat(classDef.decorators()).isEmpty();

    classDef = parse("class clazz: pass", treeMaker::classDefStatement);
    assertThat(classDef.leftPar()).isNull();
    assertThat(classDef.rightPar()).isNull();
    assertThat(classDef.args()).isNull();

    classDef = parse("class clazz(): pass", treeMaker::classDefStatement);
    assertThat(classDef.leftPar().value()).isEqualTo("(");
    assertThat(classDef.rightPar().value()).isEqualTo(")");
    assertThat(classDef.args()).isNull();

    astNode = p.parse("@foo.bar\nclass clazz: pass");
    classDef = treeMaker.classDefStatement(astNode);
    assertThat(classDef.decorators()).hasSize(1);
    Decorator decorator = classDef.decorators().get(0);

    astNode = p.parse("class clazz:\n  def foo(): pass");
    classDef = treeMaker.classDefStatement(astNode);
    FunctionDef funcDef = (FunctionDef) classDef.body().statements().get(0);
    assertThat(funcDef.isMethodDefinition()).isTrue();

    astNode = p.parse("class clazz:\n  if True:\n    def foo(): pass");
    classDef = treeMaker.classDefStatement(astNode);
    funcDef = (FunctionDef) ((IfStatement) classDef.body().statements().get(0)).body().statements().get(0);
    assertThat(funcDef.isMethodDefinition()).isTrue();

    astNode = p.parse("class ClassWithDocstring:\n" +
      "\t\"\"\"This is a docstring\"\"\"\n" +
      "\tpass");
    classDef = treeMaker.classDefStatement(astNode);
    assertThat(classDef.docstring().stringElements()).hasSize(1);
    assertThat(classDef.docstring().stringElements().get(0).value()).isEqualTo("\"\"\"This is a docstring\"\"\"");
    assertThat(classDef.children()).hasSize(7);
  }

  @Test
  void classdef_statement_type_params() {
    setRootRule(PythonGrammar.CLASSDEF);
    var classDef = parse("class generic_class[\n" +
      "   SimpleTypeVar,\n" +
      "   TypeVarWithBound: int,\n" +
      "   TypeVarWithConstraints: (str, bytes),\n" +
      "   *SimpleTypeVarTuple,\n" +
      "   **SimpleParamSpec\n" +
      "]: pass", treeMaker::classDefStatement);
    assertThat(classDef.name()).isNotNull();
    var typeParams = classDef.typeParams();
    validateTypeParams(typeParams, classDef);
  }

  @Test
  void classdef_with_expresssion_args() {
    setRootRule(PythonGrammar.CLASSDEF);
    var astNode = p.parse("""
      class name_2((name_4 for name_5 in name_0 if name_3), name_2 if name_3 else name_0):
        pass
      """);
    var classDefNode = treeMaker.classDefStatement(astNode);

    assertThat(classDefNode.name().name()).isEqualTo("name_2");

    var args = classDefNode.args().arguments();
    assertThat(args).hasSize(2);

    assertThat(args.get(0)).isInstanceOfSatisfying(RegularArgument.class, arg -> {
      assertThat(arg.expression()).isInstanceOf(ComprehensionExpression.class);
    });

    assertThat(args.get(1)).isInstanceOfSatisfying(RegularArgument.class, arg -> {
      assertThat(arg.expression()).isInstanceOf(ConditionalExpression.class);
    });
  }

  @Test
  void for_statement() {
    setRootRule(PythonGrammar.FOR_STMT);
    AstNode astNode = p.parse("for foo in bar: pass");
    ForStatement forStatement = treeMaker.forStatement(astNode);
    assertThat(forStatement.expressions()).hasSize(1);
    assertThat(forStatement.testExpressions()).hasSize(1);
    assertThat(forStatement.body().statements()).hasSize(1);
    assertThat(forStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(forStatement.elseClause()).isNull();
    assertThat(forStatement.isAsync()).isFalse();
    assertThat(forStatement.asyncKeyword()).isNull();
    assertThat(forStatement.children()).hasSize(6);
    long nbNewline = forStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.NEWLINE)).count();
    long nbIndent = forStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.INDENT)).count();
    long nbDedent = forStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.DEDENT)).count();
    assertThat(nbNewline).isEqualTo(0);
    assertThat(nbIndent).isEqualTo(0);
    assertThat(nbDedent).isEqualTo(0);

    astNode = p.parse("for foo in bar:\n  pass\n");
    forStatement = treeMaker.forStatement(astNode);
    assertThat(forStatement.expressions()).hasSize(1);
    assertThat(forStatement.testExpressions()).hasSize(1);
    assertThat(forStatement.body().statements()).hasSize(1);
    assertThat(forStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(forStatement.elseClause()).isNull();
    assertThat(forStatement.children()).hasSize(9);
    nbNewline = forStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.NEWLINE)).count();
    nbIndent = forStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.INDENT)).count();
    nbDedent = forStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.DEDENT)).count();
    assertThat(nbNewline).isEqualTo(1);
    assertThat(nbIndent).isEqualTo(1);
    assertThat(nbDedent).isEqualTo(1);

    astNode = p.parse("for foo in bar:\n  pass\nelse:\n  pass");
    forStatement = treeMaker.forStatement(astNode);
    assertThat(forStatement.expressions()).hasSize(1);
    assertThat(forStatement.testExpressions()).hasSize(1);
    assertThat(forStatement.body().statements()).hasSize(1);
    assertThat(forStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(forStatement.elseClause().body().statements()).hasSize(1);
    assertThat(forStatement.elseClause().body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(forStatement.children()).hasSize(10);
    nbNewline = forStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.NEWLINE)).count();
    nbIndent = forStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.INDENT)).count();
    nbDedent = forStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.DEDENT)).count();
    assertThat(nbNewline).isEqualTo(1);
    assertThat(nbIndent).isEqualTo(1);
    assertThat(nbDedent).isEqualTo(1);

    assertThat(forStatement.forKeyword().value()).isEqualTo("for");
    assertThat(forStatement.inKeyword().value()).isEqualTo("in");
    assertThat(forStatement.colon().value()).isEqualTo(":");
    assertThat(forStatement.elseClause().elseKeyword().value()).isEqualTo("else");

    astNode = p.parse("for foo in *x, *y, *z: pass");
    forStatement = treeMaker.forStatement(astNode);
    assertThat(forStatement.expressions()).hasSize(1);
    assertThat(forStatement.testExpressions()).extracting(Tree::getKind).containsOnly(Kind.UNPACKING_EXPR, Kind.UNPACKING_EXPR, Kind.UNPACKING_EXPR);
  }

  @Test
  void while_statement() {
    setRootRule(PythonGrammar.WHILE_STMT);
    AstNode astNode = p.parse("while foo : pass");
    WhileStatementImpl whileStatement = treeMaker.whileStatement(astNode);
    assertThat(whileStatement.condition()).isNotNull();
    assertThat(whileStatement.body().statements()).hasSize(1);
    assertThat(whileStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.elseClause()).isNull();
    assertThat(whileStatement.children()).hasSize(4);
    long nbNewline = whileStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.NEWLINE)).count();
    long nbIndent = whileStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.INDENT)).count();
    long nbDedent = whileStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.DEDENT)).count();
    assertThat(nbNewline).isEqualTo(0);
    assertThat(nbIndent).isEqualTo(0);
    assertThat(nbDedent).isEqualTo(0);

    astNode = p.parse("while foo:\n  pass\n");
    whileStatement = treeMaker.whileStatement(astNode);
    assertThat(whileStatement.condition()).isNotNull();
    assertThat(whileStatement.body().statements()).hasSize(1);
    assertThat(whileStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.elseClause()).isNull();
    assertThat(whileStatement.children()).hasSize(7);
    nbNewline = whileStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.NEWLINE)).count();
    nbIndent = whileStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.INDENT)).count();
    nbDedent = whileStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.DEDENT)).count();
    assertThat(nbNewline).isEqualTo(1);
    assertThat(nbIndent).isEqualTo(1);
    assertThat(nbDedent).isEqualTo(1);

    astNode = p.parse("while foo:\n  pass\nelse:\n  pass");
    whileStatement = treeMaker.whileStatement(astNode);
    assertThat(whileStatement.condition()).isNotNull();
    assertThat(whileStatement.body().statements()).hasSize(1);
    assertThat(whileStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.elseClause().body().statements()).hasSize(1);
    assertThat(whileStatement.elseClause().body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.children()).hasSize(8);
    nbNewline = whileStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.NEWLINE)).count();
    nbIndent = whileStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.INDENT)).count();
    nbDedent = whileStatement.children().stream().filter(c -> c.is(Tree.Kind.TOKEN) && ((Token) c).type().equals(PythonTokenType.DEDENT)).count();
    assertThat(nbNewline).isEqualTo(1);
    assertThat(nbIndent).isEqualTo(1);
    assertThat(nbDedent).isEqualTo(1);

    assertThat(whileStatement.whileKeyword().value()).isEqualTo("while");
    assertThat(whileStatement.colon().value()).isEqualTo(":");
    assertThat(whileStatement.elseClause().elseKeyword().value()).isEqualTo("else");
  }

  @Test
  void expression_statement() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("'foo'");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ExpressionStatement expressionStatement = treeMaker.expressionStatement(statementWithSeparator);
    assertThat(expressionStatement.expressions()).hasSize(1);
    assertThat(expressionStatement.children()).hasSize(1);

    astNode = p.parse("'foo', 'bar'");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    expressionStatement = treeMaker.expressionStatement(statementWithSeparator);
    assertThat(expressionStatement.expressions()).hasSize(2);
    assertThat(expressionStatement.children()).hasSize(2);
  }

  @Test
  void assignement_statement() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x = y");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AssignmentStatement pyAssignmentStatement = treeMaker.assignment(statementWithSeparator);
    assertThat(pyAssignmentStatement.firstToken().value()).isEqualTo("x");
    assertThat(pyAssignmentStatement.lastToken().value()).isEqualTo("y");
    Name assigned = (Name) pyAssignmentStatement.assignedValue();
    Name lhs = (Name) pyAssignmentStatement.lhsExpressions().get(0).expressions().get(0);
    assertThat(assigned.name()).isEqualTo("y");
    assertThat(lhs.name()).isEqualTo("x");
    assertThat(pyAssignmentStatement.children()).hasSize(3);

    astNode = p.parse("x = y = z");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    pyAssignmentStatement = treeMaker.assignment(statementWithSeparator);
    assertThat(pyAssignmentStatement.equalTokens()).hasSize(2);
    assertThat(pyAssignmentStatement.children()).hasSize(5);
    assigned = (Name) pyAssignmentStatement.assignedValue();
    lhs = (Name) pyAssignmentStatement.lhsExpressions().get(0).expressions().get(0);
    Name lhs2 = (Name) pyAssignmentStatement.lhsExpressions().get(1).expressions().get(0);
    assertThat(assigned.name()).isEqualTo("z");
    assertThat(lhs.name()).isEqualTo("x");
    assertThat(lhs2.name()).isEqualTo("y");

    astNode = p.parse("a,b = x");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    pyAssignmentStatement = treeMaker.assignment(statementWithSeparator);
    assertThat(pyAssignmentStatement.children()).hasSize(3);
    assigned = (Name) pyAssignmentStatement.assignedValue();
    ExpressionList expressionList = pyAssignmentStatement.lhsExpressions().get(0);
    assertThat(expressionList.children()).hasSize(3);
    assertThat(expressionList.children().get(1)).isSameAs(expressionList.commas().get(0));
    List<Expression> expressions = expressionList.expressions();
    assertThat(assigned.name()).isEqualTo("x");
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(expressions.get(1).getKind()).isEqualTo(Tree.Kind.NAME);

    astNode = p.parse("a,b, = x");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    pyAssignmentStatement = treeMaker.assignment(statementWithSeparator);
    assertThat(pyAssignmentStatement.children()).hasSize(3);
    assigned = (Name) pyAssignmentStatement.assignedValue();
    expressionList = pyAssignmentStatement.lhsExpressions().get(0);
    assertThat(expressionList.children()).hasSize(4);
    assertThat(expressionList.children().get(1)).isSameAs(expressionList.commas().get(0));
    assertThat(expressionList.children().get(3)).isSameAs(expressionList.commas().get(1));
    expressions = expressionList.expressions();
    assertThat(assigned.name()).isEqualTo("x");
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(expressions.get(1).getKind()).isEqualTo(Tree.Kind.NAME);

    astNode = p.parse("x = a,b");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    pyAssignmentStatement = treeMaker.assignment(statementWithSeparator);
    assertThat(pyAssignmentStatement.children()).hasSize(3);
    expressions = pyAssignmentStatement.lhsExpressions().get(0).expressions();
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyAssignmentStatement.assignedValue().getKind()).isEqualTo(Tree.Kind.TUPLE);

    astNode = p.parse("x = yield 1");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    pyAssignmentStatement = treeMaker.assignment(statementWithSeparator);
    assertThat(pyAssignmentStatement.children()).hasSize(3);
    expressions = pyAssignmentStatement.lhsExpressions().get(0).expressions();
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyAssignmentStatement.assignedValue().getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);

    // FIXME: lhs expression list shouldn't allow yield expressions. We need to change the grammar
    astNode = p.parse("x = yield 1 = y");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    pyAssignmentStatement = treeMaker.assignment(statementWithSeparator);
    assertThat(pyAssignmentStatement.children()).hasSize(5);
    List<ExpressionList> lhsExpressions = pyAssignmentStatement.lhsExpressions();
    assertThat(lhsExpressions.get(1).expressions().get(0).getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);
    assertThat(pyAssignmentStatement.assignedValue().getKind()).isEqualTo(Tree.Kind.NAME);
  }

  @Test
  void annotated_assignment() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x : string = 1");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AnnotatedAssignment annAssign = treeMaker.annotatedAssignment(statementWithSeparator);
    assertThat(annAssign.firstToken().value()).isEqualTo("x");
    assertThat(annAssign.lastToken().value()).isEqualTo("1");
    assertThat(annAssign.getKind()).isEqualTo(Tree.Kind.ANNOTATED_ASSIGNMENT);
    assertThat(annAssign.children()).hasSize(4);
    assertThat(annAssign.variable().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((Name) annAssign.variable()).name()).isEqualTo("x");
    assertThat(annAssign.assignedValue().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(annAssign.equalToken().value()).isEqualTo("=");
    assertThat(annAssign.annotation().expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(annAssign.annotation().getKind()).isEqualTo(Tree.Kind.VARIABLE_TYPE_ANNOTATION);
    assertThat(((Name) annAssign.annotation().expression()).name()).isEqualTo("string");

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    astNode = p.parse("x : string");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    annAssign = treeMaker.annotatedAssignment(statementWithSeparator);
    assertThat(annAssign.variable().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((Name) annAssign.variable()).name()).isEqualTo("x");
    assertThat(annAssign.annotation().expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(annAssign.annotation().getKind()).isEqualTo(Tree.Kind.VARIABLE_TYPE_ANNOTATION);
    assertThat(((Name) annAssign.annotation().expression()).name()).isEqualTo("string");
    assertThat(annAssign.assignedValue()).isNull();
    assertThat(annAssign.equalToken()).isNull();

    astNode = p.parse("y: int = yield _bar(x)");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    annAssign = treeMaker.annotatedAssignment(statementWithSeparator);
    assertThat(annAssign.variable().is(Kind.NAME)).isTrue();
    assertThat(((Name) annAssign.variable()).name()).isEqualTo("y");
    assertThat(annAssign.assignedValue().is(Kind.YIELD_EXPR)).isTrue();
    assertThat(((YieldExpression) annAssign.assignedValue()).expressions().size()).isEqualTo(1);
    assertThat(((YieldExpression) annAssign.assignedValue()).expressions().get(0).is(Kind.CALL_EXPR)).isTrue();

    astNode = p.parse("y: Tuple = a, b, c");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    annAssign = treeMaker.annotatedAssignment(statementWithSeparator);
    assertThat(annAssign.variable().is(Kind.NAME)).isTrue();
    assertThat(((Name) annAssign.variable()).name()).isEqualTo("y");
    assertThat(annAssign.assignedValue().is(Kind.TUPLE)).isTrue();
    assertThat(((Tuple) annAssign.assignedValue()).elements().size()).isEqualTo(3);
  }

  @Test
  void assignement_expression() {
    setRootRule(PythonGrammar.NAMED_EXPR_TEST);
    AstNode astNode = p.parse("b := 12");
    Expression expression = treeMaker.expression(astNode);
    assertThat(expression.is(Kind.ASSIGNMENT_EXPRESSION)).isTrue();
    AssignmentExpression assignmentExpression = (AssignmentExpression) expression;
    Name name = assignmentExpression.lhsName();
    Token walrus = assignmentExpression.operator();
    Expression walrusExpression = assignmentExpression.expression();
    assertThat(name.name()).isEqualTo("b");
    assertThat(walrus.value()).isEqualTo(":=");
    assertThat(walrusExpression.is(Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(((NumericLiteral) walrusExpression).valueAsString()).isEqualTo("12");

    assertThat(assignmentExpression.children()).containsExactly(name, walrus, walrusExpression);

    setRootRule(PythonGrammar.IF_STMT);
    astNode = p.parse("if a or (b := foo()):\n" +
                                      "  print(b)");
    IfStatement ifStatement = treeMaker.ifStatement(astNode);
    BinaryExpression condition = (BinaryExpression) ifStatement.condition();
    ParenthesizedExpression parenthesized = ((ParenthesizedExpression) condition.rightOperand());

    assignmentExpression = (AssignmentExpression) parenthesized.expression();

    name = assignmentExpression.lhsName();
    walrus = assignmentExpression.operator();
    walrusExpression = assignmentExpression.expression();

    assertThat(name.name()).isEqualTo("b");
    assertThat(walrus.value()).isEqualTo(":=");
    assertThat(walrusExpression.is(Kind.CALL_EXPR)).isTrue();
    Expression callee = ((CallExpression) walrusExpression).callee();
    assertThat(callee.is(Kind.NAME)).isTrue();
    assertThat(((Name) callee).name()).isEqualTo("foo");

    assertThat(assignmentExpression.children()).containsExactly(name, walrus, walrusExpression);

    setRootRule(PythonGrammar.EXPR);
    astNode = p.parse("foo(a:=42)");
    expression = treeMaker.expression(astNode);
    assertThat(expression.is(Kind.CALL_EXPR)).isTrue();
    Expression argumentExpression = ((RegularArgument) ((CallExpression) expression).arguments().get(0)).expression();
    assertThat(argumentExpression.is(Kind.ASSIGNMENT_EXPRESSION)).isTrue();
    assignmentExpression = (AssignmentExpression) argumentExpression;
    assertThat(assignmentExpression.lhsName().name()).isEqualTo("a");
    assertThat(assignmentExpression.expression().is(Kind.NUMERIC_LITERAL)).isTrue();


    setRootRule(PythonGrammar.NAMED_EXPR_TEST);
    try {
      astNode = p.parse("a.b := 12");
      treeMaker.expression(astNode);
      fail("Expected RecognitionException");
    } catch (RecognitionException e) {
      assertThat(e.getLine()).isEqualTo(1);
    }
  }

  @Test
  void compound_assignement_statement() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x += y");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    CompoundAssignmentStatement pyCompoundAssignmentStatement = treeMaker.compoundAssignment(statementWithSeparator);
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.children()).hasSize(3);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().value()).isEqualTo("+=");
    assertThat(pyCompoundAssignmentStatement.lhsExpression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyCompoundAssignmentStatement.rhsExpression().getKind()).isEqualTo(Tree.Kind.NAME);

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    astNode = p.parse("x,y,z += 1");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    pyCompoundAssignmentStatement = treeMaker.compoundAssignment(statementWithSeparator);
    assertThat(pyCompoundAssignmentStatement.firstToken().value()).isEqualTo("x");
    assertThat(pyCompoundAssignmentStatement.lastToken().value()).isEqualTo("1");
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.children()).hasSize(3);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().value()).isEqualTo("+=");
    assertThat(pyCompoundAssignmentStatement.lhsExpression().getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(pyCompoundAssignmentStatement.rhsExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    astNode = p.parse("x += yield y");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    pyCompoundAssignmentStatement = treeMaker.compoundAssignment(statementWithSeparator);
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.children()).hasSize(3);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().value()).isEqualTo("+=");
    assertThat(pyCompoundAssignmentStatement.lhsExpression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyCompoundAssignmentStatement.rhsExpression().getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);

    astNode = p.parse("x *= z");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    pyCompoundAssignmentStatement = treeMaker.compoundAssignment(statementWithSeparator);
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().value()).isEqualTo("*=");
  }

  @Test
  void try_statement() {
    setRootRule(PythonGrammar.TRY_STMT);
    AstNode astNode = p.parse("try: pass\nexcept Error: pass");
    TryStatement tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.firstToken().value()).isEqualTo("try");
    assertThat(tryStatement.lastToken().value()).isEqualTo("pass");
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.body().statements()).hasSize(1);
    assertThat(tryStatement.elseClause()).isNull();
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    assertThat(tryStatement.exceptClauses().get(0).getKind()).isEqualTo(Kind.EXCEPT_CLAUSE);
    assertThat(tryStatement.exceptClauses().get(0).firstToken().value()).isEqualTo("except");
    assertThat(tryStatement.exceptClauses().get(0).lastToken().value()).isEqualTo("pass");
    assertThat(tryStatement.exceptClauses().get(0).exceptKeyword().value()).isEqualTo("except");
    assertThat(tryStatement.exceptClauses().get(0).body().statements()).hasSize(1);
    assertThat(tryStatement.children()).hasSize(4);


    astNode = p.parse("try: pass\nexcept Error: pass\nexcept Error: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.elseClause()).isNull();
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.exceptClauses()).hasSize(2);
    assertThat(tryStatement.children()).hasSize(5);

    astNode = p.parse("try: pass\nexcept Error: pass\nfinally: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.elseClause()).isNull();
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    assertThat(tryStatement.finallyClause()).isNotNull();
    assertThat(tryStatement.finallyClause().firstToken().value()).isEqualTo("finally");
    assertThat(tryStatement.finallyClause().lastToken().value()).isEqualTo("pass");
    assertThat(tryStatement.finallyClause().finallyKeyword().value()).isEqualTo("finally");
    assertThat(tryStatement.finallyClause().body().statements()).hasSize(1);
    assertThat(tryStatement.children()).hasSize(5);

    astNode = p.parse("try: pass\nexcept Error: pass\nelse: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.elseClause().elseKeyword().value()).isEqualTo("else");
    assertThat(tryStatement.elseClause().firstToken().value()).isEqualTo("else");
    assertThat(tryStatement.elseClause().lastToken().value()).isEqualTo("pass");
    assertThat(tryStatement.elseClause().body().statements()).hasSize(1);
    assertThat(tryStatement.children()).hasSize(5);

    astNode = p.parse("try: pass\nexcept Error as e: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    ExceptClause exceptClause = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause.asKeyword().value()).isEqualTo("as");
    assertThat(exceptClause.commaToken()).isNull();
    assertThat(exceptClause.exceptionInstance()).isNotNull();
    assertThat(tryStatement.children()).hasSize(4);

    astNode = p.parse("try: pass\nexcept Error, e: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    exceptClause = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause.asKeyword()).isNull();
    assertThat(exceptClause.commaToken().value()).isEqualTo(",");
    assertThat(exceptClause.exceptionInstance()).isNotNull();
    assertThat(exceptClause.children())
      .containsExactly(exceptClause.exceptKeyword(), exceptClause.exception(), exceptClause.commaToken(), exceptClause.exceptionInstance(),
        /*colon token is not accessible through API*/ exceptClause.children().get(4), exceptClause.body());
    assertThat(tryStatement.children()).hasSize(4);

    astNode = p.parse("try:\n    pass\nexcept Error:\n    pass\nelse:\n    pass\nfinally:\n    pass");
    tryStatement = treeMaker.tryStatement(astNode);
    List<Tree> children = tryStatement.children();
    assertThat(children.get(children.size() - 2)).isSameAs(tryStatement.elseClause());
    assertThat(children.get(children.size() - 1)).isSameAs(tryStatement.finallyClause());

  }

  @Test
  void async_statement() {
    setRootRule(PythonGrammar.ASYNC_STMT);
    AstNode astNode = p.parse("async for foo in bar: pass");
    ForStatement pyForStatementTree = new PythonTreeMaker().forStatement(astNode);
    assertThat(pyForStatementTree.isAsync()).isTrue();
    assertThat(pyForStatementTree.asyncKeyword().value()).isEqualTo("async");
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body().statements()).hasSize(1);
    assertThat(pyForStatementTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseClause()).isNull();
    assertThat(pyForStatementTree.children()).hasSize(7);

    WithStatement withStatement = parse("async with foo : pass", treeMaker::withStatement);
    assertThat(withStatement.isAsync()).isTrue();
    assertThat(withStatement.asyncKeyword().value()).isEqualTo("async");
    WithItem withItem = withStatement.withItems().get(0);
    assertThat(withItem.test()).isNotNull();
    assertThat(withItem.as()).isNull();
    assertThat(withItem.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(5);
  }

  @Test
  void with_statement() {
    setRootRule(PythonGrammar.WITH_STMT);
    WithStatement withStatement = parse("with foo : pass", treeMaker::withStatement);
    assertThat(withStatement.firstToken().value()).isEqualTo("with");
    assertThat(withStatement.lastToken().value()).isEqualTo("pass");
    assertThat(withStatement.isAsync()).isFalse();
    assertThat(withStatement.asyncKeyword()).isNull();
    assertThat(withStatement.withItems()).hasSize(1);
    WithItem withItem = withStatement.withItems().get(0);
    assertThat(withItem.test()).isNotNull();
    assertThat(withItem.as()).isNull();
    assertThat(withItem.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(4);


    withStatement = parse("with foo as bar, qix :\n pass", treeMaker::withStatement);
    assertThat(withStatement.withItems()).hasSize(2);
    withItem = withStatement.withItems().get(0);
    assertThat(withItem.firstToken().value()).isEqualTo("foo");
    assertThat(withItem.lastToken().value()).isEqualTo("bar");
    assertThat(withItem.test()).isNotNull();
    assertThat(withItem.as()).isNotNull();
    assertThat(withItem.expression()).isNotNull();
    withItem = withStatement.withItems().get(1);
    assertThat(withItem.test()).isNotNull();
    assertThat(withItem.firstToken().value()).isEqualTo("qix");
    assertThat(withItem.lastToken().value()).isEqualTo("qix");
    assertThat(withItem.as()).isNull();
    assertThat(withItem.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(9);
  };

  @Test
  void with_stmt_parenthesized_context_manager() {
    setRootRule(PythonGrammar.WITH_STMT);
    WithStatement withStatement = parse(
      "with (open(\"a_really_long_foo\") as foo,\n" +
      "      open(\"a_really_long_bar\") as bar):\n" +
      "        pass", treeMaker::withStatement);
    assertThat(withStatement.children())
      .filteredOn(t -> t.is(Kind.TOKEN))
      .extracting(t -> ((Token) t).value()).contains("(", ")");
  }

  @Test
  void verify_expected_expression() {
    Map<String, Class<? extends Tree>> testData = new HashMap<>();
    testData.put("foo", Name.class);
    testData.put("foo.bar", QualifiedExpression.class);
    testData.put("foo()", CallExpression.class);
    testData.put("lambda x: x", LambdaExpression.class);

    testData.forEach((c, clazz) -> {
      FileInput pyTree = parse(c, treeMaker::fileInput);
      assertThat(pyTree.statements().statements()).hasSize(1);
      ExpressionStatement expressionStmt = (ExpressionStatement) pyTree.statements().statements().get(0);
      assertThat(expressionStmt).as(c).isInstanceOf(ExpressionStatement.class);
      assertThat(expressionStmt.expressions().get(0)).as(c).isInstanceOf(clazz);
    });
  }

  @Test
  void call_expression() {
    setRootRule(PythonGrammar.EXPR);
    CallExpression callExpression = (CallExpression) parse("foo()", treeMaker::expression);
    ArgList argList = callExpression.argumentList();
    assertThat(argList).isNull();
    assertThat(callExpression.firstToken().value()).isEqualTo("foo");
    assertThat(callExpression.lastToken().value()).isEqualTo(")");
    assertThat(callExpression.arguments()).isEmpty();
    Name name = (Name) callExpression.callee();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(3);
    assertThat(callExpression.leftPar().value()).isEqualTo("(");
    assertThat(callExpression.rightPar().value()).isEqualTo(")");

    callExpression = (CallExpression) parse("foo(x, y)", treeMaker::expression);
    argList = callExpression.argumentList();
    assertThat(argList.children()).hasSize(3);
    assertThat(argList.children().get(0)).isEqualTo(argList.arguments().get(0));
    assertThat(((Token) argList.children().get(1)).value()).isEqualTo(",");
    assertThat(argList.children().get(2)).isEqualTo(argList.arguments().get(1));
    assertThat(argList.arguments()).hasSize(2);
    assertThat(callExpression.arguments()).hasSize(2);
    Name firstArg = (Name) ((RegularArgument) argList.arguments().get(0)).expression();
    Name sndArg = (Name) ((RegularArgument) argList.arguments().get(1)).expression();
    assertThat(firstArg.name()).isEqualTo("x");
    assertThat(sndArg.name()).isEqualTo("y");
    name = (Name) callExpression.callee();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(4);

    callExpression = (CallExpression) parse("foo.bar()", treeMaker::expression);
    argList = callExpression.argumentList();
    assertThat(argList).isNull();
    QualifiedExpression callee = (QualifiedExpression) callExpression.callee();
    assertThat(callExpression.firstToken().value()).isEqualTo("foo");
    assertThat(callExpression.lastToken().value()).isEqualTo(")");
    assertThat(callee.name().name()).isEqualTo("bar");
    assertThat(((Name) callee.qualifier()).name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(3);

    assertCallExpression("func()");
    assertCallExpression("func(1,2)");
    assertCallExpression("func(*1,2)");
    assertCallExpression("func(1,**2)");
    assertCallExpression("func(value, parameter = value)");
    assertCallExpression("a.func(value)");
    assertCallExpression("a.b(value)");
    assertCallExpression("a[2](value)");
  }

  private void assertCallExpression(String code) {
    setRootRule(PythonGrammar.TEST);
    assertThat(parse(code, treeMaker::expression)).isInstanceOf(CallExpression.class);
  }

  @Test
  void combinations_with_call_expressions() {
    setRootRule(PythonGrammar.TEST);

    CallExpression nestingCall = (CallExpression) parse("foo('a').bar(42)", treeMaker::expression);
    assertThat(nestingCall.argumentList().arguments()).extracting(t -> ((RegularArgument) t).expression().getKind()).containsExactly(Tree.Kind.NUMERIC_LITERAL);
    QualifiedExpression callee = (QualifiedExpression) nestingCall.callee();
    assertThat(callee.name().name()).isEqualTo("bar");
    assertThat(callee.qualifier().firstToken().value()).isEqualTo("foo");
    assertThat(callee.qualifier().lastToken().value()).isEqualTo(")");
    assertThat(callee.qualifier().getKind()).isEqualTo(Tree.Kind.CALL_EXPR);

    nestingCall = (CallExpression) parse("foo('a').bar()", treeMaker::expression);
    assertThat(nestingCall.argumentList()).isNull();

    CallExpression callOnSubscription = (CallExpression) parse("a[42](arg)", treeMaker::expression);
    SubscriptionExpression subscription = (SubscriptionExpression) callOnSubscription.callee();
    assertThat(((Name) subscription.object()).name()).isEqualTo("a");
    assertThat(subscription.subscripts().expressions()).extracting(Tree::getKind).containsExactly(Tree.Kind.NUMERIC_LITERAL);
    assertThat(((Name) ((RegularArgument) callOnSubscription.argumentList().arguments().get(0)).expression()).name()).isEqualTo("arg");
  }

  @Test
  void attributeRef_expression() {
    setRootRule(PythonGrammar.TEST);
    QualifiedExpression qualifiedExpression = (QualifiedExpression) parse("foo.bar", treeMaker::expression);
    assertThat(qualifiedExpression.name().name()).isEqualTo("bar");
    Expression qualifier = qualifiedExpression.qualifier();
    assertThat(qualifier).isInstanceOf(Name.class);
    assertThat(((Name) qualifier).name()).isEqualTo("foo");
    assertThat(qualifiedExpression.children()).hasSize(3);
    assertThat(((Name) qualifiedExpression.children().get(0)).name()).isEqualTo("foo");
    assertThat(((Token) qualifiedExpression.children().get(1)).type()).isEqualTo(PythonPunctuator.DOT);
    assertThat(((Name) qualifiedExpression.children().get(2)).name()).isEqualTo("bar");

    qualifiedExpression = (QualifiedExpression) parse("foo.bar.baz", treeMaker::expression);
    assertThat(qualifiedExpression.name().name()).isEqualTo("baz");
    assertThat(qualifiedExpression.firstToken().value()).isEqualTo("foo");
    assertThat(qualifiedExpression.lastToken().value()).isEqualTo("baz");
    assertThat(qualifiedExpression.qualifier()).isInstanceOf(QualifiedExpression.class);
    QualifiedExpression qualExpr = (QualifiedExpression) qualifiedExpression.qualifier();
    assertThat(qualExpr.name().name()).isEqualTo("bar");
    assertThat(qualExpr.firstToken().value()).isEqualTo("foo");
    assertThat(qualExpr.lastToken().value()).isEqualTo("bar");
    assertThat(qualExpr.qualifier()).isInstanceOf(Name.class);
    Name name = (Name) qualExpr.qualifier();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(qualifiedExpression.children()).hasSize(3);
  }

  @Test
  void argument() {
    setRootRule(PythonGrammar.ARGUMENT);
    RegularArgument argumentTree = (RegularArgument) parse("foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNull();
    assertThat(argumentTree.keywordArgument()).isNull();
    Name name = (Name) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.children()).hasSize(1);

    UnpackingExpression iterableUnpacking = (UnpackingExpression) parse("*foo", treeMaker::argument);
    name = (Name) iterableUnpacking.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(iterableUnpacking.children()).hasSize(2);

    UnpackingExpression dictionaryUnpacking = (UnpackingExpression) parse("**foo", treeMaker::argument);
    name = (Name) dictionaryUnpacking.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(dictionaryUnpacking.starToken()).isNotNull();
    assertThat(dictionaryUnpacking.children()).hasSize(2);

    argumentTree = (RegularArgument) parse("bar=foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNotNull();
    Name keywordArgument = argumentTree.keywordArgument();
    assertThat(keywordArgument.name()).isEqualTo("bar");
    name = (Name) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.children()).hasSize(3).containsExactly(argumentTree.keywordArgument(), argumentTree.equalToken(), argumentTree.expression());
  }

  @Test
  void binary_expressions() {
    setRootRule(PythonGrammar.TEST);

    BinaryExpression simplePlus = binaryExpression("a + b");
    assertThat(simplePlus.leftOperand()).isInstanceOf(Name.class);
    assertThat(simplePlus.operator().value()).isEqualTo("+");
    assertThat(simplePlus.rightOperand()).isInstanceOf(Name.class);
    assertThat(simplePlus.getKind()).isEqualTo(Tree.Kind.PLUS);
    assertThat(simplePlus.children()).hasSize(3);

    BinaryExpression compoundPlus = binaryExpression("a + b - c");
    assertThat(compoundPlus.leftOperand()).isInstanceOf(BinaryExpression.class);
    assertThat(compoundPlus.children()).hasSize(3);
    assertThat(compoundPlus.operator().value()).isEqualTo("-");
    assertThat(compoundPlus.rightOperand()).isInstanceOf(Name.class);
    assertThat(compoundPlus.getKind()).isEqualTo(Tree.Kind.MINUS);
    BinaryExpression compoundPlusLeft = (BinaryExpression) compoundPlus.leftOperand();
    assertThat(compoundPlusLeft.operator().value()).isEqualTo("+");

    assertThat(binaryExpression("a * b").getKind()).isEqualTo(Tree.Kind.MULTIPLICATION);
    assertThat(binaryExpression("a / b").getKind()).isEqualTo(Tree.Kind.DIVISION);
    assertThat(binaryExpression("a // b").getKind()).isEqualTo(Tree.Kind.FLOOR_DIVISION);
    assertThat(binaryExpression("a % b").getKind()).isEqualTo(Tree.Kind.MODULO);
    assertThat(binaryExpression("a @ b").getKind()).isEqualTo(Tree.Kind.MATRIX_MULTIPLICATION);
    assertThat(binaryExpression("a >> b").getKind()).isEqualTo(Tree.Kind.SHIFT_EXPR);
    assertThat(binaryExpression("a << b").getKind()).isEqualTo(Tree.Kind.SHIFT_EXPR);
    assertThat(binaryExpression("a & b").getKind()).isEqualTo(Tree.Kind.BITWISE_AND);
    assertThat(binaryExpression("a | b").getKind()).isEqualTo(Tree.Kind.BITWISE_OR);
    assertThat(binaryExpression("a ^ b").getKind()).isEqualTo(Tree.Kind.BITWISE_XOR);
    assertThat(binaryExpression("a ** b").getKind()).isEqualTo(Tree.Kind.POWER);

    assertThat(binaryExpression("a == b").getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(binaryExpression("a >= b").getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(binaryExpression("a <= b").getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(binaryExpression("a > b").getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(binaryExpression("a < b").getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(binaryExpression("a != b").getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(binaryExpression("a <> b").getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(binaryExpression("a and b").getKind()).isEqualTo(Tree.Kind.AND);
    assertThat(binaryExpression("a or b").getKind()).isEqualTo(Tree.Kind.OR);
  }

  private BinaryExpression binaryExpression(String code) {
    Expression exp = parse(code, treeMaker::expression);
    assertThat(exp).isInstanceOf(BinaryExpression.class);
    return (BinaryExpression) exp;
  }

  @Test
  void in_expressions() {
    setRootRule(PythonGrammar.TEST);

    InExpression in = (InExpression) binaryExpression("1 in [a]");
    assertThat(in.getKind()).isEqualTo(Tree.Kind.IN);
    assertThat(in.operator().value()).isEqualTo("in");
    assertThat(in.leftOperand().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(in.rightOperand().getKind()).isEqualTo(Tree.Kind.LIST_LITERAL);
    assertThat(in.notToken()).isNull();

    InExpression notIn = (InExpression) binaryExpression("1 not in [a]");
    assertThat(notIn.getKind()).isEqualTo(Tree.Kind.IN);
    assertThat(notIn.operator().value()).isEqualTo("in");
    assertThat(notIn.notToken()).isNotNull();
    assertThat(notIn.children()).containsExactly(notIn.leftOperand(), notIn.notToken(), notIn.operator(), notIn.rightOperand());
  }

  @Test
  void is_expressions() {
    setRootRule(PythonGrammar.TEST);

    IsExpression is = (IsExpression) binaryExpression("a is 1");
    assertThat(is.getKind()).isEqualTo(Tree.Kind.IS);
    assertThat(is.operator().value()).isEqualTo("is");
    assertThat(is.leftOperand().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(is.rightOperand().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(is.notToken()).isNull();

    IsExpression notIs = (IsExpression) binaryExpression("a is not 1");
    assertThat(notIs.getKind()).isEqualTo(Tree.Kind.IS);
    assertThat(notIs.operator().value()).isEqualTo("is");
    assertThat(notIs.notToken()).isNotNull();
    assertThat(notIs.children()).containsExactly(notIs.leftOperand(), notIs.operator(), notIs.notToken(), notIs.rightOperand());
  }

  @Test
  void starred_expression() {
    setRootRule(PythonGrammar.STAR_EXPR);
    UnpackingExpression starred = (UnpackingExpression) parse("*a", treeMaker::expression);
    assertThat(starred.getKind()).isEqualTo(Tree.Kind.UNPACKING_EXPR);
    assertThat(starred.starToken().value()).isEqualTo("*");
    assertThat(starred.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(starred.children()).hasSize(2);
  }

  @Test
  void await_expression() {
    setRootRule(PythonGrammar.TEST);
    AwaitExpression expr = (AwaitExpression) parse("await x", treeMaker::expression);
    assertThat(expr.getKind()).isEqualTo(Tree.Kind.AWAIT);
    assertThat(expr.awaitToken().value()).isEqualTo("await");
    assertThat(expr.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(expr.children()).hasSize(2);

    BinaryExpression awaitWithPower = binaryExpression("await a ** 3");
    assertThat(awaitWithPower.getKind()).isEqualTo(Tree.Kind.POWER);
    assertThat(awaitWithPower.leftOperand().getKind()).isEqualTo(Tree.Kind.AWAIT);
    assertThat(awaitWithPower.rightOperand().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
  }

  @Test
  void subscription_expressions() {
    setRootRule(PythonGrammar.TEST);

    SubscriptionExpression expr = (SubscriptionExpression) parse("x[a]", treeMaker::expression);
    assertThat(expr.getKind()).isEqualTo(Tree.Kind.SUBSCRIPTION);
    assertThat(((Name) expr.object()).name()).isEqualTo("x");
    assertThat(((Name) expr.subscripts().expressions().get(0)).name()).isEqualTo("a");
    assertThat(expr.leftBracket().value()).isEqualTo("[");
    assertThat(expr.rightBracket().value()).isEqualTo("]");
    assertThat(expr.children()).hasSize(4);

    SubscriptionExpression multipleSubscripts = (SubscriptionExpression) parse("x[a, 42]", treeMaker::expression);
    ExpressionList subscripts = multipleSubscripts.subscripts();
    assertThat(subscripts.children()).hasSize(3);
    assertThat(subscripts.children().get(1)).isSameAs(subscripts.commas().get(0));
    assertThat(subscripts.expressions()).extracting(Tree::getKind)
      .containsExactly(Tree.Kind.NAME, Tree.Kind.NUMERIC_LITERAL);

    SubscriptionExpression subscriptionExpression = (SubscriptionExpression) parse("a[b:=1]", treeMaker::expression);
    assertThat(subscriptionExpression.subscripts().expressions()).hasSize(1);
    AssignmentExpression assignmentExpression = ((AssignmentExpression) subscriptionExpression.subscripts().expressions().get(0));
    assertThat(assignmentExpression.lhsName().name()).isEqualTo("b");
    assertThat(((NumericLiteral) assignmentExpression.expression()).valueAsLong()).isEqualTo(1);
  }

  @Test
  void subscription_expressions_with_unpacking_expr_subscript() {
    setRootRule(PythonGrammar.TEST);

    SubscriptionExpression expr = (SubscriptionExpression) parse("x[*a]", treeMaker::expression);
    assertThat(expr.getKind()).isEqualTo(Tree.Kind.SUBSCRIPTION);
    assertThat(((Name) expr.object()).name()).isEqualTo("x");
    assertThat(((Name) ((UnpackingExpression) expr.subscripts().expressions().get(0)).expression()).name()).isEqualTo("a");
    assertThat(expr.leftBracket().value()).isEqualTo("[");
    assertThat(expr.rightBracket().value()).isEqualTo("]");
    assertThat(expr.children()).hasSize(4);
  }

  @Test
  void slice_expressions() {
    setRootRule(PythonGrammar.TEST);

    SliceExpression expr = (SliceExpression) parse("x[a:b:c]", treeMaker::expression);
    assertThat(expr.getKind()).isEqualTo(Tree.Kind.SLICE_EXPR);
    assertThat(expr.object().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(expr.leftBracket().value()).isEqualTo("[");
    assertThat(expr.rightBracket().value()).isEqualTo("]");
    assertThat(expr.children()).hasSize(4);
    assertThat(expr.sliceList().getKind()).isEqualTo(Tree.Kind.SLICE_LIST);
    assertThat(expr.sliceList().children()).hasSize(1);
    assertThat(expr.sliceList().slices().get(0).getKind()).isEqualTo(Tree.Kind.SLICE_ITEM);

    SliceExpression multipleSlices = (SliceExpression) parse("x[a, b:c, :]", treeMaker::expression);
    List<Tree> slices = multipleSlices.sliceList().slices();
    assertThat(slices).extracting(Tree::getKind).containsExactly(Tree.Kind.NAME, Tree.Kind.SLICE_ITEM, Tree.Kind.SLICE_ITEM);
    assertThat(multipleSlices.sliceList().separators()).extracting(Token::value).containsExactly(",", ",");
  }

  @Test
  void qualified_with_slice() {
    setRootRule(PythonGrammar.TEST);
    QualifiedExpression qualifiedWithSlice = (QualifiedExpression) parse("x[a:b].foo", treeMaker::expression);
    assertThat(qualifiedWithSlice.qualifier().getKind()).isEqualTo(Tree.Kind.SLICE_EXPR);
  }

  @Test
  void slice() {
    setRootRule(PythonGrammar.SUBSCRIPT);

    SliceItem slice = parse("a:b:c", treeMaker::sliceItem);
    assertThat(((Name) slice.lowerBound()).name()).isEqualTo("a");
    assertThat(((Name) slice.upperBound()).name()).isEqualTo("b");
    assertThat(((Name) slice.stride()).name()).isEqualTo("c");
    assertThat(slice.boundSeparator().value()).isEqualTo(":");
    assertThat(slice.strideSeparator().value()).isEqualTo(":");
    assertThat(slice.children()).hasSize(5);

    SliceItem trivial = parse(":", treeMaker::sliceItem);
    assertThat(trivial.lowerBound()).isNull();
    assertThat(trivial.upperBound()).isNull();
    assertThat(trivial.stride()).isNull();
    assertThat(trivial.strideSeparator()).isNull();

    SliceItem lowerBoundOnly = parse("a:", treeMaker::sliceItem);
    assertThat(((Name) lowerBoundOnly.lowerBound()).name()).isEqualTo("a");
    assertThat(lowerBoundOnly.upperBound()).isNull();
    assertThat(lowerBoundOnly.stride()).isNull();
    assertThat(lowerBoundOnly.strideSeparator()).isNull();

    SliceItem upperBoundOnly = parse(":a", treeMaker::sliceItem);
    assertThat(upperBoundOnly.lowerBound()).isNull();
    assertThat(((Name) upperBoundOnly.upperBound()).name()).isEqualTo("a");
    assertThat(upperBoundOnly.stride()).isNull();
    assertThat(upperBoundOnly.strideSeparator()).isNull();

    SliceItem strideOnly = parse("::a", treeMaker::sliceItem);
    assertThat(strideOnly.lowerBound()).isNull();
    assertThat(strideOnly.upperBound()).isNull();
    assertThat(((Name) strideOnly.stride()).name()).isEqualTo("a");
    assertThat(strideOnly.strideSeparator()).isNotNull();

    SliceItem strideContainingOnlyColon = parse("::", treeMaker::sliceItem);
    assertThat(strideContainingOnlyColon.lowerBound()).isNull();
    assertThat(strideContainingOnlyColon.upperBound()).isNull();
    assertThat(strideContainingOnlyColon.strideSeparator()).isNotNull();
  }

  @Test
  void lambda_expr() {
    setRootRule(PythonGrammar.LAMBDEF);
    LambdaExpression lambdaExpressionTree = parse("lambda x: x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.expression()).isInstanceOf(Name.class);
    assertThat(lambdaExpressionTree.lambdaKeyword().value()).isEqualTo("lambda");
    assertThat(lambdaExpressionTree.colonToken().value()).isEqualTo(":");
    assertThat(lambdaExpressionTree.children()).doesNotContainNull();
    assertThat(lambdaExpressionTree.children()).containsExactly(lambdaExpressionTree.lambdaKeyword(),
      lambdaExpressionTree.parameters(), lambdaExpressionTree.colonToken(), lambdaExpressionTree.expression());

    assertThat(lambdaExpressionTree.parameters().nonTuple()).hasSize(1);
    assertThat(lambdaExpressionTree.children()).hasSize(4);
    assertThat(lambdaExpressionTree.children()).doesNotContainNull();

    lambdaExpressionTree = parse("lambda x, y: x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.parameters().nonTuple()).hasSize(2);
    assertThat(lambdaExpressionTree.children()).hasSize(4);
    assertThat(lambdaExpressionTree.children()).doesNotContainNull();
    assertThat(lambdaExpressionTree.children()).containsExactly(lambdaExpressionTree.lambdaKeyword(),
      lambdaExpressionTree.parameters(), lambdaExpressionTree.colonToken(), lambdaExpressionTree.expression());

    lambdaExpressionTree = parse("lambda x = 'foo': x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.parameters().all()).extracting(Tree::getKind).containsExactly(Tree.Kind.PARAMETER);
    assertThat(lambdaExpressionTree.parameters().nonTuple().get(0).name().name()).isEqualTo("x");
    assertThat(lambdaExpressionTree.children()).hasSize(4);
    assertThat(lambdaExpressionTree.children()).doesNotContainNull();
    assertThat(lambdaExpressionTree.children()).containsExactly(lambdaExpressionTree.lambdaKeyword(),
      lambdaExpressionTree.parameters(), lambdaExpressionTree.colonToken(), lambdaExpressionTree.expression());

    lambdaExpressionTree = parse("lambda (x, y): x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.parameters().all()).extracting(Tree::getKind).containsExactly(Tree.Kind.TUPLE_PARAMETER);
    assertThat(((TupleParameter) lambdaExpressionTree.parameters().all().get(0)).parameters()).hasSize(2);
    assertThat(lambdaExpressionTree.children()).hasSize(4);
    assertThat(lambdaExpressionTree.children()).doesNotContainNull();
    assertThat(lambdaExpressionTree.children()).containsExactly(lambdaExpressionTree.lambdaKeyword(),
      lambdaExpressionTree.parameters(), lambdaExpressionTree.colonToken(), lambdaExpressionTree.expression());

    lambdaExpressionTree = parse("lambda *a, **b: 42", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.parameters().nonTuple()).hasSize(2);
    Parameter starArg = lambdaExpressionTree.parameters().nonTuple().get(0);
    assertThat(starArg.starToken().value()).isEqualTo("*");
    assertThat(starArg.name().name()).isEqualTo("a");
    Parameter starStarArg = lambdaExpressionTree.parameters().nonTuple().get(1);
    assertThat(starStarArg.starToken().value()).isEqualTo("**");
    assertThat(starStarArg.name().name()).isEqualTo("b");

    lambdaExpressionTree = parse("lambda x: x if x > 1 else 0", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.getKind()).isEqualTo(Tree.Kind.LAMBDA);
    assertThat(lambdaExpressionTree.expression()).isInstanceOf(ConditionalExpression.class);

    setRootRule(PythonGrammar.LAMBDEF_NOCOND);
    lambdaExpressionTree = parse("lambda x: x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.getKind()).isEqualTo(Tree.Kind.LAMBDA);
    assertThat(lambdaExpressionTree.expression()).isInstanceOf(Name.class);

    assertThat(lambda("lambda x, *args: x").parameters().all()).hasSize(2);
    assertThat(lambda("lambda x, **kwargs: x").parameters().all()).hasSize(2);
    assertThat(lambda("lambda x, *args, **kwargs: x").parameters().all()).hasSize(3);
    assertThat(lambda("lambda x, *: x").parameters().all()).hasSize(2);
    assertThat(lambda("lambda x, /: x").parameters().all()).hasSize(2);
    assertThat(lambda("lambda x, /, *args, **kwargs: x").parameters().all()).hasSize(4);
    assertThat(lambda("lambda *, x: x").parameters().all()).hasSize(2);
    assertThat(lambda("lambda **kwargs: kwargs").parameters().all()).hasSize(1);
    assertThat(lambda("lambda x, y,: x + y").parameters().all()).hasSize(2);
  }

  private LambdaExpression lambda(String code) {
    setRootRule(PythonGrammar.LAMBDEF);
    return parse(code, treeMaker::lambdaExpression);
  }

  @Test
  void numeric_literal_expression() {
    testNumericLiteral("0", 0L);
    testNumericLiteral("12", 12L);
    testNumericLiteral("12L", 12L);
    testNumericLiteral("3_0", 30L);
    testNumericLiteral("0b01", 1L);
    testNumericLiteral("0B01", 1L);
    testNumericLiteral("0B01", 1L);
    testNumericLiteral("0B101", 5L);
    testNumericLiteral("0o777", 511L);
    testNumericLiteral("0O777", 511L);
    testNumericLiteral("0777", 511L);
    testNumericLiteral("0x28", 0x28L);
    testNumericLiteral("0X28", 0x28L);
  }

  private void testNumericLiteral(String code, Long expectedValue) {
    setRootRule(PythonGrammar.ATOM);
    Expression expression = parse(code, treeMaker::expression);
    assertThat(expression.is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    NumericLiteral numericLiteral = (NumericLiteral) expression;
    assertThat(numericLiteral.valueAsLong()).isEqualTo(expectedValue);
    assertThat(numericLiteral.valueAsString()).isEqualTo(code);
    assertThat(numericLiteral.children()).hasSize(1);
  }

  @Test
  void string_literal_expression() {
    setRootRule(PythonGrammar.ATOM);
    assertStringLiteral("''", "");
    assertStringLiteral("'\"'", "\"");
    assertStringLiteral("'\"\"\"\"\"'", "\"\"\"\"\"");
    assertStringLiteral("\"plop\"", "plop");
    assertStringLiteral("u\'plop\'", "plop", "u");
    assertStringLiteral("b\"abcdef\"", "abcdef", "b");
    assertStringLiteral("f\"\"\"Eric Idle\"\"\"", "Eric Idle", "f");
    assertStringLiteral("fr'x={4*10}'", "x={4*10}", "fr");
    assertStringLiteral("f'He said his name is {name} and he is {age} years old.'", "He said his name is {name} and he is {age} years old.", "f");
    assertStringLiteral("f'''He said his name is {name.upper()}\n    ...    and he is {6 * seven} years old.'''",
      "He said his name is {name.upper()}\n    ...    and he is {6 * seven} years old.", "f");
    assertStringLiteral("f'some: {f\"nested interpolation: {u}\"}'", "some: {f\"nested interpolation: {u}\"}", "f");
    assertThat(((StringLiteral) parse("'ab' 'cd'", treeMaker::expression)).trimmedQuotesValue()).isEqualTo("abcd");
  }

  @Test
  void string_interpolation() {
    setRootRule(PythonGrammar.ATOM);
    Expression expr = parseInterpolated("{x}").expression();
    assertThat(expr.is(Tree.Kind.NAME)).isTrue();
    assertThat(((Name) expr).name()).isEqualTo("x");

    expr = parseInterpolated("{x if p != 0 else 0}").expression();
    assertThat(expr.is(Kind.CONDITIONAL_EXPR)).isTrue();
    assertThat(((ConditionalExpression) expr).trueExpression().is(Kind.NAME)).isTrue();
    assertThat(((ConditionalExpression) expr).falseExpression().is(Kind.NUMERIC_LITERAL)).isTrue();

    expr = parseInterpolated("{\"}\" + value6}").expression();
    assertThat(expr.is(Tree.Kind.PLUS)).isTrue();
    expr = parseInterpolated("{\"}}\" + value6}").expression();
    assertThat(expr.is(Tree.Kind.PLUS)).isTrue();
    expr = parseInterpolated("{{{\"}\" + value6}}}").expression();
    assertThat(expr.is(Tree.Kind.PLUS)).isTrue();

    Expression exp = parse("F'{{bar}}'", treeMaker::expression);
    StringLiteral stringLiteral = (StringLiteral) exp;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    StringElement elmt = stringLiteral.stringElements().get(0);
    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).isEmpty();

    exp = parse("f'Some nested {f\"string \\ \n" +
                      "interpolation {x}\"}'", treeMaker::expression);
    stringLiteral = (StringLiteral) exp;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    elmt = stringLiteral.stringElements().get(0);
    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    StringLiteral interpolation = (StringLiteral) elmt.formattedExpressions().get(0).expression();
    StringElement stringElement = interpolation.stringElements().get(0);
    assertThat(stringElement.isInterpolated()).isTrue();
    Expression nestedInterpolation = stringElement.formattedExpressions().get(0).expression();
    assertThat(nestedInterpolation.is(Tree.Kind.NAME)).isTrue();
    assertThat(((Name) nestedInterpolation).name()).isEqualTo("x");
    assertThat(nestedInterpolation.firstToken().line()).isEqualTo(2);
    assertThat(nestedInterpolation.firstToken().column()).isEqualTo(15);
    // interpolated expression contains curly braces
    stringLiteral = (StringLiteral) parse("f'{ {element for element in [1, 2]} }'", treeMaker::expression);
    assertThat(stringLiteral.stringElements()).hasSize(1);
    elmt = stringLiteral.stringElements().get(0);
    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    assertThat(elmt.formattedExpressions().get(0).expression().is(Kind.SET_COMPREHENSION)).isTrue();

    stringLiteral = (StringLiteral) parse("f\"{x if p != 0 else 0}\"", treeMaker::expression);
    assertThat(stringLiteral.stringElements()).hasSize(1);
    elmt = stringLiteral.stringElements().get(0);
    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    assertThat(elmt.formattedExpressions().get(0).expression().is(Kind.CONDITIONAL_EXPR)).isTrue();

    stringLiteral = (StringLiteral) parse("f\"{x # a comment\"\n}\"", treeMaker::expression);
    assertThat(stringLiteral.stringElements()).hasSize(1);
    elmt = stringLiteral.stringElements().get(0);
    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    assertThat(elmt.formattedExpressions().get(0).expression().is(Kind.NAME)).isTrue();

    Token passToken = elmt.formattedExpressions().get(0).lastToken(); // TODO shouldn't it be the first token?
    assertThat(passToken.trivia()).hasSize(1);
    Trivia trivia = passToken.trivia().get(0);
    assertThat(trivia.token().value()).isEqualTo("# a comment\"");
    assertThat(trivia.value()).isEqualTo("# a comment\"");
    assertThat(elmt.formattedExpressions().get(0).expression().is(Kind.NAME)).isTrue();


    stringLiteral = (StringLiteral) parse("f\"\"\"foo\"\"\"", treeMaker::expression);
    assertThat(stringLiteral.stringElements()).hasSize(1);
    StringElementImpl fString = (StringElementImpl) stringLiteral.stringElements().get(0);
    assertThat(fString.isInterpolated()).isTrue();
    assertThat(fString.isTripleQuoted()).isTrue();
    assertThat(fString.formattedExpressions()).isEmpty();
    assertThat(fString.contentStartIndex()).isEqualTo(4);
    assertThat(fString.children()).hasSize(3);
    StringElementImpl fStringMiddle = (StringElementImpl) fString.children().get(1);
    assertThat(fStringMiddle.isTripleQuoted()).isFalse();
    assertThat(fStringMiddle.trimmedQuotesValue()).isEqualTo("foo");
    assertThat(fStringMiddle.contentStartIndex()).isZero();

    stringLiteral = (StringLiteral) parse("f\"foo\"", treeMaker::expression);
    assertThat(stringLiteral.stringElements()).hasSize(1);
    fString = (StringElementImpl) stringLiteral.stringElements().get(0);
    assertThat(fString.isInterpolated()).isTrue();
    assertThat(fString.isTripleQuoted()).isFalse();
    assertThat(fString.contentStartIndex()).isEqualTo(2);
    
    // Python error: f-string expression part cannot include a backslash
    assertThatThrownBy( () -> parse("f'name:\\n{na\\\nme}'", treeMaker::expression)).isInstanceOf(RecognitionException.class);
  }

  @Test
  void string_interpolation_equal_specifier() {
    setRootRule(PythonGrammar.ATOM);
    Expression exp = parse("F'{bar=}'", treeMaker::expression);
    StringLiteral stringLiteral = (StringLiteral) exp;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    StringElement elmt = stringLiteral.stringElements().get(0);

    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    FormattedExpression formattedExpression = elmt.formattedExpressions().get(0);
    assertThat(formattedExpression.expression().is(Kind.NAME)).isTrue();
    Token equalToken = formattedExpression.equalToken();
    assertThat(equalToken).isNotNull();
    assertThat(equalToken.value()).isEqualTo("=");

    exp = parse("F'{foo=} and {bar}'", treeMaker::expression);
    stringLiteral = (StringLiteral) exp;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    elmt = stringLiteral.stringElements().get(0);

    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(2);

    FormattedExpression formattedFoo = elmt.formattedExpressions().get(0);
    assertThat(formattedFoo.expression().is(Kind.NAME)).isTrue();
    assertThat(formattedFoo.equalToken()).isNotNull();

    FormattedExpression formattedBar = elmt.formattedExpressions().get(1);
    assertThat(formattedBar.expression().is(Kind.NAME)).isTrue();
    assertThat(formattedBar.equalToken()).isNull();
  }

  @Test
  void string_interpolation_nested_expressions_in_format_specifier() {
    setRootRule(PythonGrammar.ATOM);
    Expression exp = parse("f'{3.1416:{width}.{prec * 5}}'", treeMaker::expression);
    StringLiteral stringLiteral = (StringLiteral) exp;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    StringElement elmt = stringLiteral.stringElements().get(0);

    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    FormattedExpression formattedExpression = elmt.formattedExpressions().get(0);
    FormatSpecifier formatSpecifier = formattedExpression.formatSpecifier();
    assertThat(formatSpecifier).isNotNull();
    assertThat(formatSpecifier.getKind()).isEqualTo(Kind.FORMAT_SPECIFIER);
    assertThat(formatSpecifier.children()).hasSize(4);
    assertThat(formatSpecifier.formatExpressions()).hasSize(2);
    assertThat(formatSpecifier.formatExpressions().get(0).expression().is(Tree.Kind.NAME)).isTrue();
    assertThat(formatSpecifier.formatExpressions().get(1).expression().is(Kind.MULTIPLICATION)).isTrue();
  }

  @Test
  void string_interpolation_yield_expression() {
    setRootRule(PythonGrammar.ATOM);
    Expression exp = parse("f'{ yield 2 }'", treeMaker::expression);
    StringLiteral stringLiteral = (StringLiteral) exp;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    StringElement elmt = stringLiteral.stringElements().get(0);

    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    FormattedExpression formattedExpression = elmt.formattedExpressions().get(0);

    assertThat(formattedExpression.expression().is(Kind.YIELD_EXPR)).isTrue();
  }

  @Test
  void string_tuple() {
    setRootRule(PythonGrammar.ATOM);
    var stringLiteral = (StringLiteral) parse("f\"a = {h,w,a == 100,foo(),(asd, asadas)}\"", treeMaker::expression);
    assertThat(stringLiteral.stringElements()).hasSize(1);
    var elmt = stringLiteral.stringElements().get(0);

    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    var formattedExpression = elmt.formattedExpressions().get(0);

    var expression = formattedExpression.expression();
    assertThat(expression).isNotNull();
    assertThat(expression.getKind()).isEqualTo(Kind.TUPLE);
    assertThat(expression.children()).hasSize(9);


    stringLiteral = (StringLiteral) parse("f\"a = {h,!r}\"", treeMaker::expression);
    assertThat(stringLiteral.stringElements()).hasSize(1);
    elmt = stringLiteral.stringElements().get(0);

    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    formattedExpression = elmt.formattedExpressions().get(0);

    expression = formattedExpression.expression();
    assertThat(expression).isNotNull();
    assertThat(expression.getKind()).isEqualTo(Kind.TUPLE);
    assertThat(expression.children()).hasSize(2);

    stringLiteral = (StringLiteral) parse("f\"a = {h!r}\"", treeMaker::expression);
    assertThat(stringLiteral.stringElements()).hasSize(1);
    elmt = stringLiteral.stringElements().get(0);

    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    formattedExpression = elmt.formattedExpressions().get(0);

    expression = formattedExpression.expression();
    assertThat(expression).isNotNull();
    assertThat(expression.getKind()).isEqualTo(Kind.NAME);
    assertThat(expression.children()).hasSize(1);
  }

  private FormattedExpression parseInterpolated(String interpolatedExpr) {
    Expression exp = parse("f'" + interpolatedExpr + "'", treeMaker::expression);
    StringLiteral stringLiteral = (StringLiteral) exp;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    StringElement elmt = stringLiteral.stringElements().get(0);
    assertThat(elmt.isInterpolated()).isTrue();
    assertThat(elmt.formattedExpressions()).hasSize(1);
    return elmt.formattedExpressions().get(0);
  }

  private void assertStringLiteral(String fullValue, String trimmedQuoteValue) {
    assertStringLiteral(fullValue, trimmedQuoteValue, "");
  }

  private void assertStringLiteral(String fullValue, String trimmedQuoteValue, String prefix) {
    Expression parse = parse(fullValue, treeMaker::expression);
    assertThat(parse.is(Tree.Kind.STRING_LITERAL)).isTrue();
    StringLiteral stringLiteral = (StringLiteral) parse;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    StringElement firstElement = stringLiteral.stringElements().get(0);
    assertThat(firstElement.value()).isEqualTo(fullValue);
    assertThat(firstElement.trimmedQuotesValue()).isEqualTo(trimmedQuoteValue);
    assertThat(firstElement.prefix()).isEqualTo(prefix);
  }

  @Test
  void multiline_string_literal_expression() {
    setRootRule(PythonGrammar.ATOM);
    Expression parse = parse("('Hello \\ ' #Noncompliant\n            'world')", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.PARENTHESIZED)).isTrue();
    ParenthesizedExpression parenthesized = (ParenthesizedExpression) parse;
    assertThat(parenthesized.expression().is(Tree.Kind.STRING_LITERAL)).isTrue();
    StringLiteral pyStringLiteralTree = (StringLiteral) parenthesized.expression();
    assertThat(pyStringLiteralTree.children()).hasSize(2);
    assertThat(pyStringLiteralTree.stringElements().size()).isEqualTo(2);
    assertThat(pyStringLiteralTree.stringElements().get(0).value()).isEqualTo("\'Hello \\ '");
    StringElement firstElement = pyStringLiteralTree.stringElements().get(0);
    StringElement secondElement = pyStringLiteralTree.stringElements().get(1);
    assertThat(secondElement.value()).isEqualTo("'world'");
    assertThat(firstElement.trimmedQuotesValue()).isEqualTo("Hello \\ ");
    assertThat(secondElement.trimmedQuotesValue()).isEqualTo("world");
  }

  @Test
  void list_literal() {
    setRootRule(PythonGrammar.ATOM);
    Expression parse = parse("[1, \"foo\"]", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.LIST_LITERAL)).isTrue();
    assertThat(parse.firstToken().value()).isEqualTo("[");
    assertThat(parse.lastToken().value()).isEqualTo("]");
    ListLiteral listLiteral = (ListLiteral) parse;
    ExpressionList expressionList = listLiteral.elements();
    assertThat(expressionList.children()).hasSize(3);
    assertThat(expressionList.children().get(1)).isSameAs(expressionList.commas().get(0));
    List<Expression> expressions = expressionList.expressions();
    assertThat(expressions).hasSize(2);
    assertThat(expressions.get(0).is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(listLiteral.leftBracket()).isNotNull();
    assertThat(listLiteral.rightBracket()).isNotNull();
    assertThat(listLiteral.children()).hasSize(3);
  }


  @Test
  void list_comprehension() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression comprehension =
      (ComprehensionExpression) parse("[x+y for x,y in [(42, 43)]]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    assertThat(comprehension.firstToken().value()).isEqualTo("[");
    assertThat(comprehension.lastToken().value()).isEqualTo("]");
    assertThat(comprehension.resultExpression().getKind()).isEqualTo(Tree.Kind.PLUS);
    assertThat(comprehension.children()).hasSize(4);
    assertThat(((Token) comprehension.children().get(0)).value()).isEqualTo("[");
    assertThat(((Token) comprehension.children().get(3)).value()).isEqualTo("]");
    ComprehensionFor forClause = comprehension.comprehensionFor();
    assertThat(forClause.firstToken().value()).isEqualTo("for");
    assertThat(forClause.lastToken().value()).isEqualTo("]");
    assertThat(forClause.getKind()).isEqualTo(Tree.Kind.COMP_FOR);
    assertThat(forClause.forToken().value()).isEqualTo("for");
    assertThat(forClause.loopExpression().getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(forClause.inToken().value()).isEqualTo("in");
    assertThat(forClause.iterable().getKind()).isEqualTo(Tree.Kind.LIST_LITERAL);
    assertThat(forClause.nestedClause()).isNull();
    assertThat(forClause.children()).hasSize(4);
  }

  @Test
  void async_list_comprehension() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression comprehension =
      (ComprehensionExpression) parse("[i async for i in aiter() if i % 2]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    ComprehensionFor forClause = comprehension.comprehensionFor();
    assertThat(forClause.firstToken()).isSameAs(forClause.asyncToken());
    assertThat(forClause.children()).hasSize(6);
  }

  @Test
  void list_comprehension_with_if() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression comprehension =
      (ComprehensionExpression) parse("[x+1 for x in [42, 43] if x%2==0]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    ComprehensionFor forClause = comprehension.comprehensionFor();
    assertThat(forClause.nestedClause().getKind()).isEqualTo(Tree.Kind.COMP_IF);
    ComprehensionIf ifClause = (ComprehensionIf) forClause.nestedClause();
    assertThat(ifClause.ifToken().value()).isEqualTo("if");
    assertThat(ifClause.condition().getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(ifClause.nestedClause()).isNull();
    assertThat(ifClause.children()).hasSize(2);
  }

  @Test
  void list_comprehension_with_nested_for() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression comprehension =
      (ComprehensionExpression) parse("[x+y for x in [42, 43] for y in ('a', 0)]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    ComprehensionFor forClause = comprehension.comprehensionFor();
    assertThat(forClause.iterable().getKind()).isEqualTo(Tree.Kind.LIST_LITERAL);
    assertThat(forClause.nestedClause().getKind()).isEqualTo(Tree.Kind.COMP_FOR);
  }

  @Test
  void parenthesized_expression() {
    setRootRule(PythonGrammar.TEST);
    ParenthesizedExpression parenthesized = (ParenthesizedExpression) parse("(42)", treeMaker::expression);
    assertThat(parenthesized.getKind()).isEqualTo(Tree.Kind.PARENTHESIZED);
    assertThat(parenthesized.children()).hasSize(3);
    assertThat(parenthesized.leftParenthesis().value()).isEqualTo("(");
    assertThat(parenthesized.rightParenthesis().value()).isEqualTo(")");
    assertThat(parenthesized.expression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);

    parenthesized = (ParenthesizedExpression) parse("(yield 42)", treeMaker::expression);
    assertThat(parenthesized.expression().getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);
  }


  @Test
  void generator_expression() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression generator = (ComprehensionExpression) parse("(x*x for x in range(10))", treeMaker::expression);
    assertThat(generator.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);
    assertThat(generator.children()).hasSize(4);
    assertThat(generator.children().get(0)).isEqualTo(generator.firstToken());
    assertThat(generator.children().get(3)).isEqualTo(generator.lastToken());
    assertThat(generator.firstToken().value()).isEqualTo("(");
    assertThat(generator.lastToken().value()).isEqualTo(")");
    assertThat(generator.resultExpression().getKind()).isEqualTo(Tree.Kind.MULTIPLICATION);
    assertThat(generator.comprehensionFor().iterable().getKind()).isEqualTo(Tree.Kind.CALL_EXPR);

    CallExpression call = (CallExpression) parse("foo(x*x for x in range(10))", treeMaker::expression);
    assertThat(call.arguments()).hasSize(1);
    Expression firstArg = ((RegularArgument) call.arguments().get(0)).expression();
    assertThat(firstArg.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);

    call = (CallExpression) parse("foo((x*x for x in range(10)))", treeMaker::expression);
    assertThat(call.arguments()).hasSize(1);
    firstArg = ((RegularArgument) call.arguments().get(0)).expression();
    assertThat(firstArg.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);

    try {
      parse("foo(1, x*x for x in range(10))", treeMaker::expression);
      fail("generator expression must be parenthesized unless it's the unique argument in arglist");
    } catch (RecognitionException re) {
      assertThat(re).hasMessage("Parse error at line 1: Generator expression must be parenthesized if not sole argument.");
    }
  }

  @Test
  void tuples() {
    Tuple empty = parseTuple("()");
    assertThat(empty.getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(empty.firstToken().value()).isEqualTo("(");
    assertThat(empty.lastToken().value()).isEqualTo(")");
    assertThat(empty.elements()).isEmpty();
    assertThat(empty.commas()).isEmpty();
    assertThat(empty.leftParenthesis().value()).isEqualTo("(");
    assertThat(empty.rightParenthesis().value()).isEqualTo(")");
    assertThat(empty.children()).hasSize(2);

    Tuple singleValue = parseTuple("(a,)");
    assertThat(singleValue.firstToken().value()).isEqualTo("(");
    assertThat(singleValue.lastToken().value()).isEqualTo(")");
    assertThat(singleValue.elements()).extracting(Tree::getKind).containsExactly(Tree.Kind.NAME);
    assertThat(singleValue.commas()).extracting(Token::value).containsExactly(",");
    assertThat(singleValue.children()).hasSize(4);

    Tuple tuple = parseTuple("(a,b)");
    assertThat(tuple.firstToken().value()).isEqualTo("(");
    assertThat(tuple.lastToken().value()).isEqualTo(")");
    assertThat(tuple.elements()).hasSize(2);
    assertThat(tuple.children()).hasSize(5);
    assertThat(tuple.children()).containsExactly(tuple.leftParenthesis(), tuple.elements().get(0), tuple.commas().get(0), tuple.elements().get(1), tuple.rightParenthesis());

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x = a,b");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AssignmentStatement assignementstatement = treeMaker.assignment(statementWithSeparator);
    assertThat(assignementstatement.assignedValue().getKind()).isEqualTo(Tree.Kind.TUPLE);
    Tuple assignedTuple = (Tuple) assignementstatement.assignedValue();
    assertThat(assignedTuple.leftParenthesis()).isNull();
    assertThat(assignedTuple.rightParenthesis()).isNull();
    assertThat(assignedTuple.children()).doesNotContainNull();
  }

  private Tuple parseTuple(String code) {
    setRootRule(PythonGrammar.TEST);
    Tuple tuple = (Tuple) parse(code, treeMaker::expression);
    return tuple;
  }

  @Test
  void unary_expression() {
    assertUnaryExpression("-", Tree.Kind.UNARY_MINUS);
    assertUnaryExpression("+", Tree.Kind.UNARY_PLUS);
    assertUnaryExpression("~", Tree.Kind.BITWISE_COMPLEMENT);
  }

  @Test
  void not() {
    setRootRule(PythonGrammar.TEST);
    Expression exp = parse("not 1", treeMaker::expression);
    assertThat(exp).isInstanceOf(UnaryExpression.class);
    assertThat(exp.getKind()).isEqualTo(Tree.Kind.NOT);
    assertThat(((UnaryExpression) exp).expression().is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
  }

  @Test
  void conditional_expression() {
    setRootRule(PythonGrammar.TEST);
    ConditionalExpression tree = (ConditionalExpression) parse("1 if condition else 2", treeMaker::expression);
    assertThat(tree.firstToken().value()).isEqualTo("1");
    assertThat(tree.lastToken().value()).isEqualTo("2");
    assertThat(tree.ifKeyword().value()).isEqualTo("if");
    assertThat(tree.elseKeyword().value()).isEqualTo("else");
    assertThat(tree.condition().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(tree.trueExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(tree.falseExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(tree.children()).containsExactly(tree.trueExpression(), tree.ifKeyword(), tree.condition(), tree.elseKeyword(), tree.falseExpression());

    ConditionalExpression nestedConditionalExpressionTree =
      (ConditionalExpression) parse("1 if x else 2 if y else 3", treeMaker::expression);
    assertThat(nestedConditionalExpressionTree.condition().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(nestedConditionalExpressionTree.trueExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    Expression nestedConditionalExpr = nestedConditionalExpressionTree.falseExpression();
    assertThat(nestedConditionalExpr.firstToken().value()).isEqualTo("2");
    assertThat(nestedConditionalExpr.lastToken().value()).isEqualTo("3");
    assertThat(nestedConditionalExpr.getKind()).isEqualTo(Tree.Kind.CONDITIONAL_EXPR);
  }

  @Test
  void dictionary_literal() {
    setRootRule(PythonGrammar.ATOM);
    DictionaryLiteral tree = (DictionaryLiteral) parse("{'key': 'value'}", treeMaker::expression);
    assertThat(tree.firstToken().value()).isEqualTo("{");
    assertThat(tree.lastToken().value()).isEqualTo("}");
    assertThat(tree.getKind()).isEqualTo(Tree.Kind.DICTIONARY_LITERAL);
    assertThat(tree.elements()).hasSize(1);
    KeyValuePair keyValuePair = (KeyValuePair) tree.elements().iterator().next();
    assertThat(keyValuePair.getKind()).isEqualTo(Tree.Kind.KEY_VALUE_PAIR);
    assertThat(keyValuePair.key().getKind()).isEqualTo(Tree.Kind.STRING_LITERAL);
    assertThat(keyValuePair.colon().value()).isEqualTo(":");
    assertThat(keyValuePair.value().getKind()).isEqualTo(Tree.Kind.STRING_LITERAL);
    assertThat(tree.children()).hasSize(3).containsExactly(tree.lCurlyBrace(), tree.elements().get(0), tree.rCurlyBrace());

    tree = (DictionaryLiteral) parse("{'key': 'value', 'key2': 'value2'}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);
    assertThat(tree.children()).hasSize(5).containsExactly(tree.lCurlyBrace(), tree.elements().get(0), tree.commas().get(0), tree.elements().get(1), tree.rCurlyBrace());

    tree = (DictionaryLiteral) parse("{** var}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(1);
    UnpackingExpression dictUnpacking = (UnpackingExpression) tree.elements().iterator().next();
    assertThat(dictUnpacking.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(dictUnpacking.starToken().value()).isEqualTo("**");

    tree = (DictionaryLiteral) parse("{** var, key: value}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);
  }

  @Test
  void dict_comprehension() {
    setRootRule(PythonGrammar.TEST);
    DictCompExpression comprehension =
      (DictCompExpression) parse("{x-1:y+1 for x,y in [(42,43)]}", treeMaker::expression);
    assertThat(comprehension.firstToken().value()).isEqualTo("{");
    assertThat(comprehension.lastToken().value()).isEqualTo("}");
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.DICT_COMPREHENSION);
    assertThat(comprehension.colonToken().value()).isEqualTo(":");
    assertThat(comprehension.keyExpression().getKind()).isEqualTo(Tree.Kind.MINUS);
    assertThat(comprehension.valueExpression().getKind()).isEqualTo(Tree.Kind.PLUS);
    assertThat(comprehension.comprehensionFor().loopExpression().getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(comprehension.children()).hasSize(6);
    assertThat(comprehension.firstToken().value()).isEqualTo("{");
    assertThat(comprehension.lastToken().value()).isEqualTo("}");
  }

  @Test
  void set_literal() {
    setRootRule(PythonGrammar.ATOM);
    SetLiteral tree = (SetLiteral) parse("{ x }", treeMaker::expression);
    assertThat(tree.firstToken().value()).isEqualTo("{");
    assertThat(tree.lastToken().value()).isEqualTo("}");
    assertThat(tree.getKind()).isEqualTo(Tree.Kind.SET_LITERAL);
    assertThat(tree.elements()).hasSize(1);
    Expression element = tree.elements().iterator().next();
    assertThat(element.getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(tree.lCurlyBrace().value()).isEqualTo("{");
    assertThat(tree.rCurlyBrace().value()).isEqualTo("}");
    assertThat(tree.commas()).hasSize(0);
    assertThat(tree.children()).hasSize(3).containsExactly(tree.lCurlyBrace(), tree.elements().get(0), tree.rCurlyBrace());

    tree = (SetLiteral) parse("{ x, y, }", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);
    assertThat(tree.children()).hasSize(6).containsExactly(tree.lCurlyBrace(), tree.elements().get(0), tree.commas().get(0),
      tree.elements().get(1), tree.commas().get(1), tree.rCurlyBrace());

    tree = (SetLiteral) parse("{ first := x, second := y }", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);
    assertThat(tree.children()).hasSize(5).containsExactly(tree.lCurlyBrace(), tree.elements().get(0), tree.commas().get(0),
      tree.elements().get(1), tree.rCurlyBrace());

    tree = (SetLiteral) parse("{ *x }", treeMaker::expression);
    assertThat(tree.elements()).hasSize(1);
    element = tree.elements().iterator().next();
    assertThat(element.getKind()).isEqualTo(Tree.Kind.UNPACKING_EXPR);
  }

  @Test
  void set_comprehension() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression comprehension =
      (ComprehensionExpression) parse("{x-1 for x in [42, 43]}", treeMaker::expression);
    assertThat(comprehension.firstToken().value()).isEqualTo("{");
    assertThat(comprehension.lastToken().value()).isEqualTo("}");
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.SET_COMPREHENSION);
    assertThat(comprehension.resultExpression().getKind()).isEqualTo(Tree.Kind.MINUS);
    assertThat(comprehension.children()).hasSize(4);
    assertThat(((Token) comprehension.children().get(0)).value()).isEqualTo("{");
    assertThat(((Token) comprehension.children().get(3)).value()).isEqualTo("}");
  }

  @Test
  void set_comprehension_with_walrus_operator() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression comprehension =
      (ComprehensionExpression) parse("{last := x-1 for x in [42, 43]}", treeMaker::expression);
    assertThat(comprehension.firstToken().value()).isEqualTo("{");
    assertThat(comprehension.lastToken().value()).isEqualTo("}");
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.SET_COMPREHENSION);
    assertThat(comprehension.resultExpression().getKind()).isEqualTo(Kind.ASSIGNMENT_EXPRESSION);
    assertThat(((AssignmentExpression) comprehension.resultExpression()).operator().value()).isEqualTo(":=");
    assertThat(comprehension.children()).hasSize(4);
    assertThat(((Token) comprehension.children().get(0)).value()).isEqualTo("{");
    assertThat(((Token) comprehension.children().get(3)).value()).isEqualTo("}");
  }

  @Test
  void repr_expression() {
    setRootRule(PythonGrammar.ATOM);
    ReprExpression reprExpressionTree = (ReprExpression) parse("`1`", treeMaker::expression);
    assertThat(reprExpressionTree.getKind()).isEqualTo(Tree.Kind.REPR);
    assertThat(reprExpressionTree.firstToken().value()).isEqualTo("`");
    assertThat(reprExpressionTree.lastToken().value()).isEqualTo("`");
    assertThat(reprExpressionTree.openingBacktick().value()).isEqualTo("`");
    assertThat(reprExpressionTree.closingBacktick().value()).isEqualTo("`");
    assertThat(reprExpressionTree.expressionList().expressions()).hasSize(1);
    assertThat(reprExpressionTree.expressionList().expressions().get(0).getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(reprExpressionTree.children()).hasSize(3);

    reprExpressionTree = (ReprExpression) parse("`x, y`", treeMaker::expression);
    assertThat(reprExpressionTree.getKind()).isEqualTo(Tree.Kind.REPR);
    assertThat(reprExpressionTree.expressionList().expressions()).hasSize(2);
    assertThat(reprExpressionTree.expressionList().children()).hasSize(3);
    assertThat(reprExpressionTree.expressionList().commas()).hasSize(1);
    assertThat(reprExpressionTree.expressionList().commas().get(0)).isSameAs(reprExpressionTree.expressionList().children().get(1));
    assertThat(reprExpressionTree.expressionList().expressions().get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(reprExpressionTree.expressionList().expressions().get(1).getKind()).isEqualTo(Tree.Kind.NAME);

  }

  @Test
  @Timeout(2)
  public void should_not_require_exponential_time() {
    try {
      p.parse("((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((");
      fail("Expected RecognitionException");
    } catch (RecognitionException e) {
      assertThat(e.getLine()).isEqualTo(1);
    }
    try {
      p.parse("````````````````````````````````````````````````````````````````````````````````");
      fail("Expected RecognitionException");
    } catch (RecognitionException e) {
      assertThat(e.getLine()).isEqualTo(1);
    }
  }

  @Test
  void ellipsis_expression() {
    setRootRule(PythonGrammar.ATOM);
    EllipsisExpression ellipsisExpressionTree = (EllipsisExpression) parse("...", treeMaker::expression);
    assertThat(ellipsisExpressionTree.getKind()).isEqualTo(Tree.Kind.ELLIPSIS);
    assertThat(ellipsisExpressionTree.ellipsis()).extracting(Token::value).containsExactly(".", ".", ".");
    assertThat(ellipsisExpressionTree.children()).hasSize(3);
  }

  @Test
  void none_expression() {
    setRootRule(PythonGrammar.ATOM);
    NoneExpression noneExpressionTree = (NoneExpression) parse("None", treeMaker::expression);
    assertThat(noneExpressionTree.getKind()).isEqualTo(Tree.Kind.NONE);
    assertThat(noneExpressionTree.none().value()).isEqualTo("None");
    assertThat(noneExpressionTree.children()).hasSize(1);
  }

  @Test
  void variables() {
    setRootRule(PythonGrammar.EXPR);
    Name name = (Name) parse("foo", treeMaker::expression);
    assertThat(name.isVariable()).isTrue();

    QualifiedExpression qualifiedExpressionTree = (QualifiedExpression) parse("a.b", treeMaker::expression);
    assertThat(qualifiedExpressionTree.name().isVariable()).isFalse();

    setRootRule(PythonGrammar.FUNCDEF);
    FunctionDef functionDefTree = parse("def func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.name().isVariable()).isFalse();
  }


  @Test
  void test_trivia() {
    FileInput fileInput = parse("#A comment\npass", treeMaker::fileInput);

    assertThat(fileInput.firstToken().value()).isEqualTo("pass");
    Token passToken = fileInput.firstToken();
    assertThat(passToken.trivia()).hasSize(1);
    Trivia trivia = passToken.trivia().get(0);
    assertThat(trivia.token().value()).isEqualTo("#A comment");
    assertThat(trivia.value()).isEqualTo("#A comment");

    fileInput = parse("foo", treeMaker::fileInput);
    assertThat(fileInput.firstToken().value()).isEqualTo("foo");
    passToken = fileInput.firstToken();
    assertThat(passToken.trivia()).hasSize(0);
  }

  @Test
  void statements_separators() {
    FileInput tree = parse("foo(); bar()\ntoto()", treeMaker::fileInput);
    List<Statement> statements = tree.statements().statements();

    List<Tree> statementChildren = statements.get(0).children();
    assertThat(statementChildren.get(statementChildren.size() - 1).is(Tree.Kind.TOKEN)).isTrue();
    Token token = (Token) statementChildren.get(statementChildren.size() - 1);
    assertThat(token.type()).isEqualTo(PythonPunctuator.SEMICOLON);

    statementChildren = statements.get(1).children();
    assertThat(statementChildren.get(statementChildren.size() - 1).is(Tree.Kind.TOKEN)).isTrue();
    token = (Token) statementChildren.get(statementChildren.size() - 1);
    assertThat(token.type()).isEqualTo(PythonTokenType.NEWLINE);

    tree = parse("foo()\ntoto()", treeMaker::fileInput);
    statements = tree.statements().statements();
    statementChildren = statements.get(0).children();
    assertThat(statementChildren.get(statementChildren.size() - 1).is(Tree.Kind.TOKEN)).isTrue();
    token = (Token) statementChildren.get(statementChildren.size() - 1);
    assertThat(token.type()).isEqualTo(PythonTokenType.NEWLINE);

    // Check that the second semicolon is not ignored
    tree = parse("foo(); bar();\ntoto()", treeMaker::fileInput);
    statements = tree.statements().statements();
    assertThat(statements.get(0).lastToken().value()).isEqualTo(")");
    statementChildren = statements.get(0).children();
    assertThat(statementChildren.get(statementChildren.size() - 1).is(Tree.Kind.TOKEN)).isTrue();
    token = (Token) statementChildren.get(statementChildren.size() - 1);
    assertThat(token.type()).isEqualTo(PythonPunctuator.SEMICOLON);

    statementChildren = statements.get(1).children();
    assertThat(statementChildren.get(statementChildren.size() - 2).is(Tree.Kind.TOKEN)).isTrue();
    token = (Token) statementChildren.get(statementChildren.size() - 2);
    assertThat(token.type()).isEqualTo(PythonPunctuator.SEMICOLON);
    assertThat(statementChildren.get(statementChildren.size() - 1).is(Tree.Kind.TOKEN)).isTrue();
    token = (Token) statementChildren.get(statementChildren.size() - 1);
    assertThat(token.type()).isEqualTo(PythonTokenType.NEWLINE);
    assertThat(statements.get(1).lastToken().value()).isEqualTo(")");

    assertThat(statements.get(2).lastToken().value()).isEqualTo(")");
  }

  @Test
  void separators() {
    List<Tree.Kind> compoundStatements = Arrays.asList(Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT, Tree.Kind.IF_STMT, Tree.Kind.ELSE_CLAUSE, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF,
      Tree.Kind.TRY_STMT, Tree.Kind.EXCEPT_CLAUSE);
    File file = new File("src/test/resources/separator.py");
    String content = fileContent(file);
    FileInput tree = parse(content, treeMaker::fileInput);
    for (Statement statement : tree.statements().statements()) {
      if (compoundStatements.contains(statement.getKind())) {
        assertThat(statement.separator()).isNull();
      } else {
        assertThat(statement.separator()).isNotNull();
        assertThat(statement.separator().type()).isEqualTo(PythonTokenType.NEWLINE);
      }
    }
  }

  private void assertUnaryExpression(String operator, Tree.Kind kind) {
    setRootRule(PythonGrammar.EXPR);
    Expression parse = parse(operator + "1", treeMaker::expression);
    assertThat(parse.is(kind)).isTrue();
    UnaryExpression unary = (UnaryExpression) parse;
    assertThat(unary.expression().is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(unary.operator().value()).isEqualTo(operator);
    assertThat(unary.children()).hasSize(2);
  }

  @Test
  void except_group() {
    FileInput tree = parse("try:pass\nexcept* OSError:pass", treeMaker::fileInput);
    TryStatement tryStatement = (TryStatement) tree.statements().statements().get(0);
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    ExceptClause exceptClause = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause.getKind()).isEqualTo(Kind.EXCEPT_GROUP_CLAUSE);
    assertThat(exceptClause.starToken()).isNotNull();
    assertThat(exceptClause.starToken().value()).isEqualTo("*");
    assertThat(exceptClause.exception().is(Kind.NAME)).isTrue();

    assertThatThrownBy(() -> parse("try:pass\nexcept* OSError:pass\nexcept IOError:pass", treeMaker::fileInput))
      .isInstanceOf(RecognitionException.class)
      .hasMessage("Parse error at line 3: Try statement cannot contain both except and except* clauses.");

    assertThatThrownBy(() -> parse("try:pass\nexcept*:pass", treeMaker::fileInput))
      .isInstanceOf(RecognitionException.class)
      .hasMessage("Parse error at line 2: except* clause must specify the type of the expected exception.");
  }

  /**
   * except* body cannot contain continue, break or return instruction
   */
  @Test
  void except_group_invalid_instruction() {
    String code1 = "try:pass\n" +
      "except* OSError:\n" +
      "  continue";
    assertThatThrownBy(() -> parse(code1, treeMaker::fileInput))
      .isInstanceOf(RecognitionException.class)
      .hasMessage("Parse error at line 3: continue statement cannot appear in except* block.");

    String code2 = "try:pass\n" +
      "except* OSError:\n" +
      "  break";
    assertThatThrownBy(() -> parse(code2, treeMaker::fileInput))
      .isInstanceOf(RecognitionException.class)
      .hasMessage("Parse error at line 3: break statement cannot appear in except* block.");

    String code3 = "try:pass\n" +
      "except* OSError:\n" +
      "  return";
    assertThatThrownBy(() -> parse(code3, treeMaker::fileInput))
      .isInstanceOf(RecognitionException.class)
      .hasMessage("Parse error at line 3: return statement cannot appear in except* block.");

    // should not return parse error if continue/break is in a loop
    String code4 = "try:pass\n" +
      "except* OSError:\n" +
      "  while true:break";
    FileInput tree = parse(code4, treeMaker::fileInput);
    TryStatement tryStatement = (TryStatement) tree.statements().statements().get(0);
    assertThat(tryStatement.exceptClauses().get(0).getKind()).isEqualTo(Kind.EXCEPT_GROUP_CLAUSE);

    String code5 = "try:pass\n" +
      "except* OSError:\n" +
      "  for x in \"bob\":continue";
    tree = parse(code5, treeMaker::fileInput);
    tryStatement = (TryStatement) tree.statements().statements().get(0);
    assertThat(tryStatement.exceptClauses().get(0).getKind()).isEqualTo(Kind.EXCEPT_GROUP_CLAUSE);

    // should not return parse error if return is in a function
    String code6 = "try:pass\n" +
      "except* OSError:\n" +
      "  def foo():return";
    tree = parse(code6, treeMaker::fileInput);
    tryStatement = (TryStatement) tree.statements().statements().get(0);
    assertThat(tryStatement.exceptClauses().get(0).getKind()).isEqualTo(Kind.EXCEPT_GROUP_CLAUSE);

    String code7 = "for x in range(42):\n" +
      "  try:pass\n" +
      "  except* OSError:\n" +
      "    continue";
    assertThatThrownBy(() -> parse(code7, treeMaker::fileInput))
      .isInstanceOf(RecognitionException.class)
      .hasMessage("Parse error at line 4: continue statement cannot appear in except* block.");

    String code8 = "def foo():\n" +
      "  try:pass\n" +
      "  except* OSError:\n" +
      "    continue";
    assertThatThrownBy(() -> parse(code8, treeMaker::fileInput))
      .isInstanceOf(RecognitionException.class)
      .hasMessage("Parse error at line 4: continue statement cannot appear in except* block.");
  }

  @Test
  void except_group_multiple() {
    FileInput tree = parse("try:pass\nexcept* OSError:pass\nexcept* ValueError:pass", treeMaker::fileInput);
    TryStatement tryStatement = (TryStatement) tree.statements().statements().get(0);
    assertThat(tryStatement.exceptClauses()).hasSize(2);

    ExceptClause exceptClause1 = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause1.getKind()).isEqualTo(Kind.EXCEPT_GROUP_CLAUSE);
    assertThat(exceptClause1.starToken()).isNotNull();
    assertThat(exceptClause1.starToken().value()).isEqualTo("*");
    assertThat(exceptClause1.exception().is(Kind.NAME)).isTrue();
    assertThat(((Name) exceptClause1.exception()).name()).isEqualTo("OSError");

    ExceptClause exceptClause2 = tryStatement.exceptClauses().get(1);
    assertThat(exceptClause2.getKind()).isEqualTo(Kind.EXCEPT_GROUP_CLAUSE);
    assertThat(exceptClause2.starToken()).isNotNull();
    assertThat(exceptClause2.starToken().value()).isEqualTo("*");
    assertThat(exceptClause2.exception().is(Kind.NAME)).isTrue();
    assertThat(((Name) exceptClause2.exception()).name()).isEqualTo("ValueError");
  }

  @Test
  void typeAliasStatement() {
    setRootRule(PythonGrammar.TYPE_ALIAS_STMT);

    var astNode = p.parse("type A[B] = str");
    var statementWithSeparator = new StatementWithSeparator(astNode, null);
    var typeAlias = treeMaker.typeAliasStatement(statementWithSeparator);

    assertThat(typeAlias).isNotNull();
    assertThat(typeAlias.typeKeyword()).isNotNull();
    assertThat(typeAlias.typeKeyword().value()).isEqualTo("type");
    assertThat(typeAlias.name()).isNotNull();
    assertThat(typeAlias.name().name()).isEqualTo("A");
    assertThat(typeAlias.typeParams()).isNotNull();
    assertThat(typeAlias.equalToken()).isNotNull();
    assertThat(typeAlias.expression()).isNotNull();
    assertThat(typeAlias.expression().is(Kind.NAME)).isTrue();
    assertThat(((Name) typeAlias.expression()).name()).isEqualTo("str");
  }

  @Test
  void typeAliasWithComprehensions() {
    setRootRule(PythonGrammar.TYPE_ALIAS_STMT);
    var astNode = p.parse("type A = [i for i in range(3)]");
    var statementWithSeparator = new StatementWithSeparator(astNode, null);
    var typeAlias = treeMaker.typeAliasStatement(statementWithSeparator);

    assertThat(typeAlias.expression()).isInstanceOf(ComprehensionExpression.class);
    ComprehensionExpression comprehensionExpr = (ComprehensionExpression) typeAlias.expression();
    assertThat(comprehensionExpr.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
  }

  public String fileContent(File file) {
    try {
      return new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new IllegalStateException("Cannot read " + file, e);
    }
  }
}
