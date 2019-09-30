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
package org.sonar.python.tree;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.RecognitionException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.api.tree.AliasedName;
import org.sonar.python.api.tree.AnnotatedAssignment;
import org.sonar.python.api.tree.Argument;
import org.sonar.python.api.tree.AssertStatement;
import org.sonar.python.api.tree.AssignmentStatement;
import org.sonar.python.api.tree.AwaitExpression;
import org.sonar.python.api.tree.BinaryExpression;
import org.sonar.python.api.tree.BreakStatement;
import org.sonar.python.api.tree.CallExpression;
import org.sonar.python.api.tree.ClassDef;
import org.sonar.python.api.tree.CompoundAssignmentStatement;
import org.sonar.python.api.tree.ComprehensionExpression;
import org.sonar.python.api.tree.ComprehensionFor;
import org.sonar.python.api.tree.ComprehensionIf;
import org.sonar.python.api.tree.ConditionalExpression;
import org.sonar.python.api.tree.ContinueStatement;
import org.sonar.python.api.tree.Decorator;
import org.sonar.python.api.tree.DelStatement;
import org.sonar.python.api.tree.DictCompExpression;
import org.sonar.python.api.tree.DictionaryLiteral;
import org.sonar.python.api.tree.EllipsisExpression;
import org.sonar.python.api.tree.ElseStatement;
import org.sonar.python.api.tree.ExceptClause;
import org.sonar.python.api.tree.ExecStatement;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.ExpressionList;
import org.sonar.python.api.tree.ExpressionStatement;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.ForStatement;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.GlobalStatement;
import org.sonar.python.api.tree.IfStatement;
import org.sonar.python.api.tree.ImportFrom;
import org.sonar.python.api.tree.ImportName;
import org.sonar.python.api.tree.ImportStatement;
import org.sonar.python.api.tree.InExpression;
import org.sonar.python.api.tree.IsExpression;
import org.sonar.python.api.tree.KeyValuePair;
import org.sonar.python.api.tree.LambdaExpression;
import org.sonar.python.api.tree.ListLiteral;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.NoneExpression;
import org.sonar.python.api.tree.NonlocalStatement;
import org.sonar.python.api.tree.NumericLiteral;
import org.sonar.python.api.tree.Parameter;
import org.sonar.python.api.tree.ParenthesizedExpression;
import org.sonar.python.api.tree.PassStatement;
import org.sonar.python.api.tree.PrintStatement;
import org.sonar.python.api.tree.QualifiedExpression;
import org.sonar.python.api.tree.RaiseStatement;
import org.sonar.python.api.tree.ReprExpression;
import org.sonar.python.api.tree.ReturnStatement;
import org.sonar.python.api.tree.SetLiteral;
import org.sonar.python.api.tree.SliceExpression;
import org.sonar.python.api.tree.SliceItem;
import org.sonar.python.api.tree.StarredExpression;
import org.sonar.python.api.tree.Statement;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.StringElement;
import org.sonar.python.api.tree.StringLiteral;
import org.sonar.python.api.tree.SubscriptionExpression;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TryStatement;
import org.sonar.python.api.tree.Tuple;
import org.sonar.python.api.tree.TupleParameter;
import org.sonar.python.api.tree.TypeAnnotation;
import org.sonar.python.api.tree.UnaryExpression;
import org.sonar.python.api.tree.WhileStatement;
import org.sonar.python.api.tree.WithItem;
import org.sonar.python.api.tree.WithStatement;
import org.sonar.python.api.tree.YieldExpression;
import org.sonar.python.api.tree.YieldStatement;
import org.sonar.python.parser.RuleTest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

public class PythonTreeMakerTest extends RuleTest {

  private final PythonTreeMaker treeMaker = new PythonTreeMaker();

  @Test
  public void fileInputTreeOnEmptyFile() {
    FileInput pyTree = parse("", treeMaker::fileInput);
    assertThat(pyTree.statements()).isNull();
    assertThat(pyTree.docstring()).isNull();

    pyTree = parse("\"\"\"\n" +
      "This is a module docstring\n" +
      "\"\"\"", treeMaker::fileInput);
    assertThat(pyTree.docstring().value()).isEqualTo("\"\"\"\n" +
      "This is a module docstring\n" +
      "\"\"\"");

    pyTree = parse("if x:\n pass", treeMaker::fileInput);
    IfStatement ifStmt = (IfStatement) pyTree.statements().statements().get(0);
    assertThat(ifStmt.body().parent()).isEqualTo(ifStmt);
  }

  @Test
  public void descendants_and_ancestors() {
    FileInput pyTree = parse("def foo(): pass\ndef bar(): pass", treeMaker::fileInput);
    assertThat(pyTree.descendants().filter(tree -> tree.getKind() != Tree.Kind.TOKEN).count()).isEqualTo(9);
    assertThat(pyTree.descendants(Tree.Kind.STATEMENT_LIST).count()).isEqualTo(3);
    assertThat(pyTree.descendants(Tree.Kind.FUNCDEF).count()).isEqualTo(2);
    assertThat(pyTree.descendants(Tree.Kind.NAME).count()).isEqualTo(2);
    assertThat(pyTree.descendants(Tree.Kind.PASS_STMT).count()).isEqualTo(2);

    FunctionDef functionDef = (FunctionDef) pyTree.descendants(Tree.Kind.FUNCDEF).collect(Collectors.toList()).get(0);
    assertThat(functionDef.ancestors()).extracting(Tree::getKind).containsExactly(Tree.Kind.STATEMENT_LIST, Tree.Kind.FILE_INPUT);

    PassStatement passStmt = (PassStatement) pyTree.descendants(Tree.Kind.PASS_STMT).collect(Collectors.toList()).get(0);
    assertThat(passStmt.ancestors()).extracting(Tree::getKind).containsExactly(
      Tree.Kind.STATEMENT_LIST, Tree.Kind.FUNCDEF, Tree.Kind.STATEMENT_LIST, Tree.Kind.FILE_INPUT);
  }

  @Test
  public void unexpected_expression_should_throw_an_exception() {
    try {
      parse("", treeMaker::expression);
      fail("unexpected ASTNode type for expression should not succeed to be translated to Strongly typed AST");
    } catch (IllegalStateException iae) {
      assertThat(iae).hasMessage("Expression FILE_INPUT not correctly translated to strongly typed AST");
    }
  }

  @Test
  public void verify_expected_statement() {
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

    testData.forEach((c,clazz) -> {
      FileInput pyTree = parse(c, treeMaker::fileInput);
      StatementList statementList = pyTree.statements();
      assertThat(statementList.statements()).hasSize(1);
      Statement stmt = statementList.statements().get(0);
      assertThat(stmt.parent()).isEqualTo(statementList);
      assertThat(stmt).as(c).isInstanceOf(clazz);
    });
  }

  @Test
  public void IfStatement() {
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
    ElseStatement elseBranch = pyIfStatementTree.elseBranch();
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

    // tokens
    AstNode parseTree = p.parse("if x: pass");
    IfStatement pyFileInputTree = treeMaker.ifStatement(parseTree);
    assertThat(pyFileInputTree.body().tokens().stream().map(Token::token)).isEqualTo(parseTree.getFirstChild(PythonGrammar.SUITE).getTokens());
  }

  @Test
  public void printStatement() {
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
  public void execStatement() {
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
    assertThat(execStatement.children()).hasSize(3);

    astNode = p.parse("exec 'foo' in globals, locals");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    execStatement = treeMaker.execStatement(statementWithSeparator);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().value()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNotNull();
    assertThat(execStatement.localsExpression()).isNotNull();
    assertThat(execStatement.children()).hasSize(4);

    // TODO: exec stmt should parse exec ('foo', globals, locals); see https://docs.python.org/2/reference/simple_stmts.html#exec
  }

  @Test
  public void assertStatement() {
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
  public void passStatement() {
    setRootRule(PythonGrammar.PASS_STMT);
    AstNode astNode = p.parse("pass");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    PassStatement passStatement = treeMaker.passStatement(statementWithSeparator);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.passKeyword().value()).isEqualTo("pass");
    assertThat(passStatement.children()).hasSize(1);
  }

  @Test
  public void delStatement() {
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
  public void returnStatement() {
    setRootRule(PythonGrammar.RETURN_STMT);
    AstNode astNode = p.parse("return foo");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ReturnStatement returnStatement = treeMaker.returnStatement(statementWithSeparator);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().value()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(1);
    assertThat(returnStatement.children()).hasSize(2);

    astNode = p.parse("return foo, bar");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    returnStatement = treeMaker.returnStatement(statementWithSeparator);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().value()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(2);
    assertThat(returnStatement.children()).hasSize(3);

    astNode = p.parse("return");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    returnStatement = treeMaker.returnStatement(statementWithSeparator);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().value()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(0);
    assertThat(returnStatement.children()).hasSize(1);
  }

  @Test
  public void yieldStatement() {
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
  }

  @Test
  public void raiseStatement() {
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
  public void breakStatement() {
    setRootRule(PythonGrammar.BREAK_STMT);
    AstNode astNode = p.parse("break");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    BreakStatement breakStatement = treeMaker.breakStatement(statementWithSeparator);
    assertThat(breakStatement).isNotNull();
    assertThat(breakStatement.breakKeyword().value()).isEqualTo("break");
    assertThat(breakStatement.children()).hasSize(1);
  }

  @Test
  public void continueStatement() {
    setRootRule(PythonGrammar.CONTINUE_STMT);
    AstNode astNode = p.parse("continue");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ContinueStatement continueStatement = treeMaker.continueStatement(statementWithSeparator);
    assertThat(continueStatement).isNotNull();
    assertThat(continueStatement.continueKeyword().value()).isEqualTo("continue");
    assertThat(continueStatement.children()).hasSize(1);
  }

  @Test
  public void importStatement() {
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
  public void importFromStatement() {
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

    astNode = p.parse("from .foo import f");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    importStatement = (ImportFrom) treeMaker.importStatement(statementWithSeparator);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).value()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.children()).hasSize(5);

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
  }

  @Test
  public void globalStatement() {
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
  public void nonlocalStatement() {
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
  public void funcdef_statement() {
    setRootRule(PythonGrammar.FUNCDEF);
    AstNode astNode = p.parse("def func(): pass");
    FunctionDef functionDefTree = treeMaker.funcDefStatement(astNode);
    assertThat(functionDefTree.name()).isNotNull();
    assertThat(functionDefTree.name().name()).isEqualTo("func");
    assertThat(functionDefTree.body().statements()).hasSize(1);
    assertThat(functionDefTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(functionDefTree.children()).hasSize(6);
    assertThat(functionDefTree.parameters()).isNull();
    assertThat(functionDefTree.isMethodDefinition()).isFalse();
    assertThat(functionDefTree.docstring()).isNull();
    assertThat(functionDefTree.decorators()).isEmpty();
    assertThat(functionDefTree.asyncKeyword()).isNull();
    assertThat(functionDefTree.returnTypeAnnotation()).isNull();
    assertThat(functionDefTree.colon().value()).isEqualTo(":");
    assertThat(functionDefTree.defKeyword().value()).isEqualTo("def");
    assertThat(functionDefTree.leftPar().value()).isEqualTo("(");
    assertThat(functionDefTree.rightPar().value()).isEqualTo(")");

    functionDefTree = parse("def func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.parameters().all()).hasSize(1);

    functionDefTree = parse("async def func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.asyncKeyword().value()).isEqualTo("async");

    functionDefTree = parse("def func(x) -> string : pass", treeMaker::funcDefStatement);
    TypeAnnotation returnType = functionDefTree.returnTypeAnnotation();
    assertThat(returnType.getKind()).isEqualTo(Tree.Kind.RETURN_TYPE_ANNOTATION);
    assertThat(returnType.dash().value()).isEqualTo("-");
    assertThat(returnType.gt().value()).isEqualTo(">");
    assertThat(returnType.expression().getKind()).isEqualTo(Tree.Kind.NAME);

    functionDefTree = parse("@foo\ndef func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.decorators()).hasSize(1);
    Decorator decoratorTree = functionDefTree.decorators().get(0);
    assertThat(decoratorTree.getKind()).isEqualTo(Tree.Kind.DECORATOR);
    assertThat(decoratorTree.atToken().value()).isEqualTo("@");
    assertThat(decoratorTree.name().names().get(0).name()).isEqualTo("foo");
    assertThat(decoratorTree.leftPar()).isNull();
    assertThat(decoratorTree.arguments()).isNull();
    assertThat(decoratorTree.rightPar()).isNull();

    functionDefTree = parse("@foo()\n@bar(1)\ndef func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.decorators()).hasSize(2);
    Decorator decoratorTree1 = functionDefTree.decorators().get(0);
    assertThat(decoratorTree1.leftPar().value()).isEqualTo("(");
    assertThat(decoratorTree1.arguments()).isNull();
    assertThat(decoratorTree1.rightPar().value()).isEqualTo(")");
    Decorator decoratorTree2 = functionDefTree.decorators().get(1);
    assertThat(decoratorTree2.arguments().arguments()).extracting(arg -> arg.expression().getKind()).containsExactly(Tree.Kind.NUMERIC_LITERAL);

    functionDefTree = parse("def func(x, y): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.parameters().all()).hasSize(2);

    functionDefTree = parse("def func(x = 'foo', y): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.parameters().all()).isEqualTo(functionDefTree.parameters().nonTuple());
    List<Parameter> parameters = functionDefTree.parameters().nonTuple();
    assertThat(parameters).hasSize(2);
    Parameter parameter1 = parameters.get(0);
    assertThat(parameter1.name().name()).isEqualTo("x");
    assertThat(parameter1.equalToken().value()).isEqualTo("=");
    assertThat(parameter1.defaultValue().is(Tree.Kind.STRING_LITERAL)).isTrue();
    Parameter parameter2 = parameters.get(1);
    assertThat(parameter2.equalToken()).isNull();
    assertThat(parameter2.defaultValue()).isNull();

    functionDefTree = parse("def func(p1, *p2, **p3): pass", treeMaker::funcDefStatement);
    parameters = functionDefTree.parameters().nonTuple();
    assertThat(parameters).extracting(p -> p.name().name()).containsExactly("p1", "p2", "p3");
    assertThat(parameters).extracting(p -> p.starToken() == null ? null : p.starToken().value()).containsExactly(null, "*", "**");

    functionDefTree = parse("def func(x : int, y): pass", treeMaker::funcDefStatement);
    parameters = functionDefTree.parameters().nonTuple();
    assertThat(parameters).hasSize(2);
    TypeAnnotation annotation = parameters.get(0).typeAnnotation();
    assertThat(annotation.getKind()).isEqualTo(Tree.Kind.TYPE_ANNOTATION);
    assertThat(annotation.colonToken().value()).isEqualTo(":");
    assertThat(((Name) annotation.expression()).name()).isEqualTo("int");
    assertThat(annotation.children()).hasSize(2);
    assertThat(parameters.get(1).typeAnnotation()).isNull();

    functionDefTree = parse("def func(a, ((b, c), d)): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.parameters().all()).hasSize(2);
    assertThat(functionDefTree.parameters().all()).extracting(Tree::getKind).containsExactly(Tree.Kind.PARAMETER, Tree.Kind.TUPLE_PARAMETER);
    TupleParameter tupleParam = (TupleParameter) functionDefTree.parameters().all().get(1);
    assertThat(tupleParam.openingParenthesis().value()).isEqualTo("(");
    assertThat(tupleParam.closingParenthesis().value()).isEqualTo(")");
    assertThat(tupleParam.parameters()).extracting(Tree::getKind).containsExactly(Tree.Kind.TUPLE_PARAMETER, Tree.Kind.PARAMETER);
    assertThat(tupleParam.commas()).extracting(Token::value).containsExactly(",");
    assertThat(tupleParam.children()).hasSize(3);

    functionDefTree = parse("def func(x : int, y):\n  \"\"\"\n" +
      "This is a function docstring\n" +
      "\"\"\"\n  pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.docstring().value()).isEqualTo("\"\"\"\n" +
      "This is a function docstring\n" +
      "\"\"\"");
  }

  @Test
  public void classdef_statement() {
    setRootRule(PythonGrammar.CLASSDEF);
    AstNode astNode = p.parse("class clazz(Parent): pass");
    ClassDef classDefTree = treeMaker.classDefStatement(astNode);
    assertThat(classDefTree.name()).isNotNull();
    assertThat(classDefTree.docstring()).isNull();
    assertThat(classDefTree.classKeyword().value()).isEqualTo("class");
    assertThat(classDefTree.leftPar().value()).isEqualTo("(");
    assertThat(classDefTree.rightPar().value()).isEqualTo(")");
    assertThat(classDefTree.colon().value()).isEqualTo(":");
    assertThat(classDefTree.name().name()).isEqualTo("clazz");
    assertThat(classDefTree.body().statements()).hasSize(1);
    assertThat(classDefTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(classDefTree.args().is(Tree.Kind.ARG_LIST)).isTrue();
    assertThat(classDefTree.args().children()).hasSize(1);
    assertThat(classDefTree.args().arguments().get(0).is(Tree.Kind.ARGUMENT)).isTrue();
    assertThat(classDefTree.decorators()).isEmpty();

    classDefTree = parse("class clazz: pass", treeMaker::classDefStatement);
    assertThat(classDefTree.leftPar()).isNull();
    assertThat(classDefTree.rightPar()).isNull();
    assertThat(classDefTree.args()).isNull();

    classDefTree = parse("class clazz(): pass", treeMaker::classDefStatement);
    assertThat(classDefTree.leftPar().value()).isEqualTo("(");
    assertThat(classDefTree.rightPar().value()).isEqualTo(")");
    assertThat(classDefTree.args()).isNull();

    astNode = p.parse("@foo.bar\nclass clazz: pass");
    classDefTree = treeMaker.classDefStatement(astNode);
    assertThat(classDefTree.decorators()).hasSize(1);
    Decorator decorator = classDefTree.decorators().get(0);
    assertThat(decorator.name().names()).extracting(Name::name).containsExactly("foo", "bar");

    astNode = p.parse("class clazz:\n  def foo(): pass");
    classDefTree = treeMaker.classDefStatement(astNode);
    FunctionDef funcDef = (FunctionDef) classDefTree.body().statements().get(0);
    assertThat(funcDef.isMethodDefinition()).isTrue();

    astNode = p.parse("class clazz:\n  if True:\n    def foo(): pass");
    classDefTree = treeMaker.classDefStatement(astNode);
    funcDef = (FunctionDef) ((IfStatement) classDefTree.body().statements().get(0)).body().statements().get(0);
    assertThat(funcDef.isMethodDefinition()).isTrue();

    astNode = p.parse("class ClassWithDocstring:\n" +
      "\t\"\"\"This is a docstring\"\"\"\n" +
      "\tpass");
    classDefTree = treeMaker.classDefStatement(astNode);
    assertThat(classDefTree.docstring().value()).isEqualTo("\"\"\"This is a docstring\"\"\"");
    assertThat(classDefTree.children()).hasSize(8);
  }

  @Test
  public void for_statement() {
    setRootRule(PythonGrammar.FOR_STMT);
    AstNode astNode = p.parse("for foo in bar: pass");
    ForStatement pyForStatementTree = treeMaker.forStatement(astNode);
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body().statements()).hasSize(1);
    assertThat(pyForStatementTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody()).isNull();
    assertThat(pyForStatementTree.isAsync()).isFalse();
    assertThat(pyForStatementTree.asyncKeyword()).isNull();
    assertThat(pyForStatementTree.children()).hasSize(7);

    astNode = p.parse("for foo in bar:\n  pass\nelse:\n  pass");
    pyForStatementTree = treeMaker.forStatement(astNode);
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body().statements()).hasSize(1);
    assertThat(pyForStatementTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody().statements()).hasSize(1);
    assertThat(pyForStatementTree.elseBody().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.children()).hasSize(15);

    assertThat(pyForStatementTree.forKeyword().value()).isEqualTo("for");
    assertThat(pyForStatementTree.inKeyword().value()).isEqualTo("in");
    assertThat(pyForStatementTree.colon().value()).isEqualTo(":");
    assertThat(pyForStatementTree.elseKeyword().value()).isEqualTo("else");
    assertThat(pyForStatementTree.elseColon().value()).isEqualTo(":");
  }

  @Test
  public void while_statement() {
    setRootRule(PythonGrammar.WHILE_STMT);
    AstNode astNode = p.parse("while foo : pass");
    WhileStatementImpl whileStatement = treeMaker.whileStatement(astNode);
    assertThat(whileStatement.condition()).isNotNull();
    assertThat(whileStatement.body().statements()).hasSize(1);
    assertThat(whileStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.elseBody()).isNull();
    assertThat(whileStatement.children()).hasSize(4);

    astNode = p.parse("while foo:\n  pass\nelse:\n  pass");
    whileStatement = treeMaker.whileStatement(astNode);
    assertThat(whileStatement.condition()).isNotNull();
    assertThat(whileStatement.body().statements()).hasSize(1);
    assertThat(whileStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.elseBody().statements()).hasSize(1);
    assertThat(whileStatement.elseBody().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.children()).hasSize(13);

    assertThat(whileStatement.whileKeyword().value()).isEqualTo("while");
    assertThat(whileStatement.colon().value()).isEqualTo(":");
    assertThat(whileStatement.elseKeyword().value()).isEqualTo("else");
    assertThat(whileStatement.elseColon().value()).isEqualTo(":");

  }

  @Test
  public void expression_statement() {
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
  public void assignement_statement() {
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
    List<Expression> expressions = pyAssignmentStatement.lhsExpressions().get(0).expressions();
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
  public void annotated_assignment() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x : string = 1");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AnnotatedAssignment annAssign = treeMaker.annotatedAssignment(statementWithSeparator);
    assertThat(annAssign.firstToken().value()).isEqualTo("x");
    assertThat(annAssign.lastToken().value()).isEqualTo("1");
    assertThat(annAssign.getKind()).isEqualTo(Tree.Kind.ANNOTATED_ASSIGNMENT);
    assertThat(annAssign.children()).hasSize(5);
    assertThat(annAssign.variable().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((Name) annAssign.variable()).name()).isEqualTo("x");
    assertThat(annAssign.assignedValue().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(annAssign.equalToken().value()).isEqualTo("=");
    assertThat(annAssign.annotation().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((Name) annAssign.annotation()).name()).isEqualTo("string");
    assertThat(annAssign.colonToken().value()).isEqualTo(":");

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    astNode = p.parse("x : string");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    annAssign = treeMaker.annotatedAssignment(statementWithSeparator);
    assertThat(annAssign.variable().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((Name) annAssign.variable()).name()).isEqualTo("x");
    assertThat(annAssign.annotation().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((Name) annAssign.annotation()).name()).isEqualTo("string");
    assertThat(annAssign.assignedValue()).isNull();
    assertThat(annAssign.equalToken()).isNull();
  }

  @Test
  public void compound_assignement_statement() {
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
  public void try_statement() {
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
    assertThat(tryStatement.children()).hasSize(4);
  }

  @Test
  public void async_statement() {
    setRootRule(PythonGrammar.ASYNC_STMT);
    AstNode astNode = p.parse("async for foo in bar: pass");
    ForStatement pyForStatementTree = new PythonTreeMaker().forStatement(astNode);
    assertThat(pyForStatementTree.isAsync()).isTrue();
    assertThat(pyForStatementTree.asyncKeyword().value()).isEqualTo("async");
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body().statements()).hasSize(1);
    assertThat(pyForStatementTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody()).isNull();
    assertThat(pyForStatementTree.children()).hasSize(8);

    WithStatement withStatement = parse("async with foo : pass", treeMaker::withStatement);
    assertThat(withStatement.isAsync()).isTrue();
    assertThat(withStatement.asyncKeyword().value()).isEqualTo("async");
    WithItem withItem = withStatement.withItems().get(0);
    assertThat(withItem.test()).isNotNull();
    assertThat(withItem.as()).isNull();
    assertThat(withItem.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(4);
  }

  @Test
  public void with_statement() {
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


    withStatement = parse("with foo as bar, qix : pass", treeMaker::withStatement);
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
    assertThat(withStatement.children()).hasSize(5);
  }

  @Test
  public void verify_expected_expression() {
    Map<String, Class<? extends Tree>> testData = new HashMap<>();
    testData.put("foo", Name.class);
    testData.put("foo.bar", QualifiedExpression.class);
    testData.put("foo()", CallExpression.class);
    testData.put("lambda x: x", LambdaExpression.class);

    testData.forEach((c,clazz) -> {
      FileInput pyTree = parse(c, treeMaker::fileInput);
      assertThat(pyTree.statements().statements()).hasSize(1);
      ExpressionStatement expressionStmt = (ExpressionStatement) pyTree.statements().statements().get(0);
      assertThat(expressionStmt).as(c).isInstanceOf(ExpressionStatement.class);
      assertThat(expressionStmt.expressions().get(0)).as(c).isInstanceOf(clazz);
    });
  }

  @Test
  public void call_expression() {
    setRootRule(PythonGrammar.CALL_EXPR);
    CallExpression callExpression = parse("foo()", treeMaker::callExpression);
    assertThat(callExpression.argumentList()).isNull();
    assertThat(callExpression.firstToken().value()).isEqualTo("foo");
    assertThat(callExpression.lastToken().value()).isEqualTo(")");
    assertThat(callExpression.arguments()).isEmpty();
    Name name = (Name) callExpression.callee();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(3);
    assertThat(callExpression.leftPar().value()).isEqualTo("(");
    assertThat(callExpression.rightPar().value()).isEqualTo(")");

    callExpression = parse("foo(x, y)", treeMaker::callExpression);
    assertThat(callExpression.argumentList().arguments()).hasSize(2);
    assertThat(callExpression.arguments()).hasSize(2);
    Name firstArg = (Name) callExpression.argumentList().arguments().get(0).expression();
    Name sndArg = (Name) callExpression.argumentList().arguments().get(1).expression();
    assertThat(firstArg.name()).isEqualTo("x");
    assertThat(sndArg.name()).isEqualTo("y");
    name = (Name) callExpression.callee();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(4);

    callExpression = parse("foo.bar()", treeMaker::callExpression);
    assertThat(callExpression.argumentList()).isNull();
    QualifiedExpression callee = (QualifiedExpression) callExpression.callee();
    assertThat(callExpression.firstToken().value()).isEqualTo("foo");
    assertThat(callExpression.lastToken().value()).isEqualTo(")");
    assertThat(callee.name().name()).isEqualTo("bar");
    assertThat(((Name) callee.qualifier()).name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(3);
  }

  @Test
  public void combinations_with_call_expressions() {
    setRootRule(PythonGrammar.TEST);

    CallExpression nestingCall = (CallExpression) parse("foo('a').bar(42)", treeMaker::expression);
    assertThat(nestingCall.argumentList().arguments()).extracting(t -> t.expression().getKind()).containsExactly(Tree.Kind.NUMERIC_LITERAL);
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
    assertThat(((Name) callOnSubscription.argumentList().arguments().get(0).expression()).name()).isEqualTo("arg");
  }

  @Test
  public void attributeRef_expression() {
    setRootRule(PythonGrammar.ATTRIBUTE_REF);
    QualifiedExpression qualifiedExpression = parse("foo.bar", treeMaker::qualifiedExpression);
    assertThat(qualifiedExpression.name().name()).isEqualTo("bar");
    Expression qualifier = qualifiedExpression.qualifier();
    assertThat(qualifier).isInstanceOf(Name.class);
    assertThat(((Name) qualifier).name()).isEqualTo("foo");
    assertThat(qualifiedExpression.children()).hasSize(3);

    qualifiedExpression = parse("foo.bar.baz", treeMaker::qualifiedExpression);
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
  public void argument() {
    setRootRule(PythonGrammar.ARGUMENT);
    Argument argumentTree = parse("foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNull();
    assertThat(argumentTree.keywordArgument()).isNull();
    Name name = (Name) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNull();
    assertThat(argumentTree.starStarToken()).isNull();
    assertThat(argumentTree.children()).hasSize(1);

    argumentTree = parse("*foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNull();
    assertThat(argumentTree.keywordArgument()).isNull();
    name = (Name) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNotNull();
    assertThat(argumentTree.starStarToken()).isNull();
    assertThat(argumentTree.children()).hasSize(2);

    argumentTree = parse("**foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNull();
    assertThat(argumentTree.keywordArgument()).isNull();
    name = (Name) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNull();
    assertThat(argumentTree.starStarToken()).isNotNull();
    assertThat(argumentTree.children()).hasSize(2);

    argumentTree = parse("bar=foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNotNull();
    Name keywordArgument = argumentTree.keywordArgument();
    assertThat(keywordArgument.name()).isEqualTo("bar");
    name = (Name) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNull();
    assertThat(argumentTree.starStarToken()).isNull();
    assertThat(argumentTree.children()).hasSize(3);
  }

  @Test
  public void binary_expressions() {
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
  public void in_expressions() {
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
  }

  @Test
  public void is_expressions() {
    setRootRule(PythonGrammar.TEST);

    IsExpression in = (IsExpression) binaryExpression("a is 1");
    assertThat(in.getKind()).isEqualTo(Tree.Kind.IS);
    assertThat(in.operator().value()).isEqualTo("is");
    assertThat(in.leftOperand().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(in.rightOperand().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(in.notToken()).isNull();

    IsExpression notIn = (IsExpression) binaryExpression("a is not 1");
    assertThat(notIn.getKind()).isEqualTo(Tree.Kind.IS);
    assertThat(notIn.operator().value()).isEqualTo("is");
    assertThat(notIn.notToken()).isNotNull();
  }

  @Test
  public void starred_expression() {
    setRootRule(PythonGrammar.STAR_EXPR);
    StarredExpression starred = (StarredExpression) parse("*a", treeMaker::expression);
    assertThat(starred.getKind()).isEqualTo(Tree.Kind.STARRED_EXPR);
    assertThat(starred.starToken().value()).isEqualTo("*");
    assertThat(starred.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(starred.children()).hasSize(2);
  }

  @Test
  public void await_expression() {
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
  public void subscription_expressions() {
    setRootRule(PythonGrammar.TEST);

    SubscriptionExpression expr = (SubscriptionExpression) parse("x[a]", treeMaker::expression);
    assertThat(expr.getKind()).isEqualTo(Tree.Kind.SUBSCRIPTION);
    assertThat(((Name) expr.object()).name()).isEqualTo("x");
    assertThat(((Name) expr.subscripts().expressions().get(0)).name()).isEqualTo("a");
    assertThat(expr.leftBracket().value()).isEqualTo("[");
    assertThat(expr.rightBracket().value()).isEqualTo("]");
    assertThat(expr.children()).hasSize(4);

    SubscriptionExpression multipleSubscripts = (SubscriptionExpression) parse("x[a, 42]", treeMaker::expression);
    assertThat(multipleSubscripts.subscripts().expressions()).extracting(Tree::getKind)
      .containsExactly(Tree.Kind.NAME, Tree.Kind.NUMERIC_LITERAL);
  }

  @Test
  public void slice_expressions() {
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
  public void qualified_with_slice() {
    setRootRule(PythonGrammar.TEST);
    QualifiedExpression qualifiedWithSlice = (QualifiedExpression) parse("x[a:b].foo", treeMaker::expression);
    assertThat(qualifiedWithSlice.qualifier().getKind()).isEqualTo(Tree.Kind.SLICE_EXPR);
  }

  @Test
  public void slice() {
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
  public void lambda_expr() {
    setRootRule(PythonGrammar.LAMBDEF);
    LambdaExpression lambdaExpressionTree = parse("lambda x: x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.expression()).isInstanceOf(Name.class);
    assertThat(lambdaExpressionTree.lambdaKeyword().value()).isEqualTo("lambda");
    assertThat(lambdaExpressionTree.colonToken().value()).isEqualTo(":");

    assertThat(lambdaExpressionTree.parameters().nonTuple()).hasSize(1);
    assertThat(lambdaExpressionTree.children()).hasSize(4);

    lambdaExpressionTree = parse("lambda x, y: x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.parameters().nonTuple()).hasSize(2);
    assertThat(lambdaExpressionTree.children()).hasSize(4);

    lambdaExpressionTree = parse("lambda x = 'foo': x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.parameters().all()).extracting(Tree::getKind).containsExactly(Tree.Kind.PARAMETER);
    assertThat(lambdaExpressionTree.parameters().nonTuple().get(0).name().name()).isEqualTo("x");
    assertThat(lambdaExpressionTree.children()).hasSize(4);

    lambdaExpressionTree = parse("lambda (x, y): x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.parameters().all()).extracting(Tree::getKind).containsExactly(Tree.Kind.TUPLE_PARAMETER);
    assertThat(((TupleParameter) lambdaExpressionTree.parameters().all().get(0)).parameters()).hasSize(2);
    assertThat(lambdaExpressionTree.children()).hasSize(4);

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
  }

  @Test
  public void numeric_literal_expression() {
    testNumericLiteral("12", 12L);
    testNumericLiteral("12L", 12L);
    testNumericLiteral("3_0", 30L);
    testNumericLiteral("0b01", 1L);
    testNumericLiteral("0B01", 1L);
    testNumericLiteral("0B01", 1L);
    testNumericLiteral("0B101", 5L);
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
  public void string_literal_expression() {
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
    assertThat(((StringLiteral) parse("'ab' 'cd'", treeMaker::expression)).trimmedQuotesValue()).isEqualTo("abcd");
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
    assertThat(firstElement.children()).hasSize(1);
  }

  @Test
  public void multiline_string_literal_expression() {
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
  public void list_literal() {
    setRootRule(PythonGrammar.ATOM);
    Expression parse = parse("[1, \"foo\"]", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.LIST_LITERAL)).isTrue();
    assertThat(parse.firstToken().value()).isEqualTo("[");
    assertThat(parse.lastToken().value()).isEqualTo("]");
    ListLiteral listLiteralTree = (ListLiteral) parse;
    List<Expression> expressions = listLiteralTree.elements().expressions();
    assertThat(expressions).hasSize(2);
    assertThat(expressions.get(0).is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(listLiteralTree.leftBracket()).isNotNull();
    assertThat(listLiteralTree.rightBracket()).isNotNull();
    assertThat(listLiteralTree.children()).hasSize(3);
  }


  @Test
  public void list_comprehension() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression comprehension =
      (ComprehensionExpression) parse("[x+y for x,y in [(42, 43)]]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    assertThat(comprehension.firstToken().value()).isEqualTo("[");
    assertThat(comprehension.lastToken().value()).isEqualTo("]");
    assertThat(comprehension.resultExpression().getKind()).isEqualTo(Tree.Kind.PLUS);
    assertThat(comprehension.children()).hasSize(2);
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
  public void list_comprehension_with_if() {
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
  public void list_comprehension_with_nested_for() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression comprehension =
      (ComprehensionExpression) parse("[x+y for x in [42, 43] for y in ('a', 0)]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    ComprehensionFor forClause = comprehension.comprehensionFor();
    assertThat(forClause.iterable().getKind()).isEqualTo(Tree.Kind.LIST_LITERAL);
    assertThat(forClause.nestedClause().getKind()).isEqualTo(Tree.Kind.COMP_FOR);
  }

  @Test
  public void parenthesized_expression() {
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
  public void generator_expression() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression generator = (ComprehensionExpression) parse("(x*x for x in range(10))", treeMaker::expression);
    assertThat(generator.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);
    assertThat(generator.children()).hasSize(2);
    assertThat(generator.firstToken().value()).isEqualTo("(");
    assertThat(generator.lastToken().value()).isEqualTo(")");
    assertThat(generator.resultExpression().getKind()).isEqualTo(Tree.Kind.MULTIPLICATION);
    assertThat(generator.comprehensionFor().iterable().getKind()).isEqualTo(Tree.Kind.CALL_EXPR);

    setRootRule(PythonGrammar.CALL_EXPR);
    CallExpression call = (CallExpression) parse("foo(x*x for x in range(10))", treeMaker::expression);
    assertThat(call.arguments()).hasSize(1);
    Expression firstArg = call.arguments().get(0).expression();
    assertThat(firstArg.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);

    call = (CallExpression) parse("foo((x*x for x in range(10)))", treeMaker::expression);
    assertThat(call.arguments()).hasSize(1);
    firstArg = call.arguments().get(0).expression();
    assertThat(firstArg.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);

    try {
      parse("foo(1, x*x for x in range(10))", treeMaker::expression);
      fail("generator expression must be parenthesized unless it's the unique argument in arglist");
    } catch (RecognitionException re) {
      assertThat(re).hasMessage("Parse error at line 1: Generator expression must be parenthesized if not sole argument.");
    }
  }

  @Test
  public void tuples() {
    Tuple empty = parseTuple("()");
    assertThat(empty.getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(empty.elements()).isEmpty();
    assertThat(empty.commas()).isEmpty();
    assertThat(empty.leftParenthesis().value()).isEqualTo("(");
    assertThat(empty.rightParenthesis().value()).isEqualTo(")");
    assertThat(empty.children()).hasSize(2);

    Tuple singleValue = parseTuple("(a,)");
    assertThat(singleValue.elements()).extracting(Tree::getKind).containsExactly(Tree.Kind.NAME);
    assertThat(singleValue.commas()).extracting(Token::value).containsExactly(",");
    assertThat(singleValue.children()).hasSize(4);

    assertThat(parseTuple("(a,b)").elements()).hasSize(2);
  }

  private Tuple parseTuple(String code) {
    setRootRule(PythonGrammar.TEST);
    Tuple tuple = (Tuple) parse(code, treeMaker::expression);
    assertThat(tuple.firstToken().value()).isEqualTo("(");
    assertThat(tuple.lastToken().value()).isEqualTo(")");
    return tuple;
  }

  @Test
  public void unary_expression() {
    assertUnaryExpression("-", Tree.Kind.UNARY_MINUS);
    assertUnaryExpression("+", Tree.Kind.UNARY_PLUS);
    assertUnaryExpression("~", Tree.Kind.BITWISE_COMPLEMENT);
  }

  @Test
  public void not() {
    setRootRule(PythonGrammar.TEST);
    Expression exp = parse("not 1", treeMaker::expression);
    assertThat(exp).isInstanceOf(UnaryExpression.class);
    assertThat(exp.getKind()).isEqualTo(Tree.Kind.NOT);
    assertThat(((UnaryExpression) exp).expression().is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
  }

  @Test
  public void conditional_expression() {
    setRootRule(PythonGrammar.TEST);
    ConditionalExpression conditionalExpressionTree = (ConditionalExpression) parse("1 if condition else 2", treeMaker::expression);
    assertThat(conditionalExpressionTree.firstToken().value()).isEqualTo("1");
    assertThat(conditionalExpressionTree.lastToken().value()).isEqualTo("2");
    assertThat(conditionalExpressionTree.ifKeyword().value()).isEqualTo("if");
    assertThat(conditionalExpressionTree.elseKeyword().value()).isEqualTo("else");
    assertThat(conditionalExpressionTree.condition().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(conditionalExpressionTree.trueExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(conditionalExpressionTree.falseExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);

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
  public void dictionary_literal() {
    setRootRule(PythonGrammar.ATOM);
    DictionaryLiteral tree = (DictionaryLiteral) parse("{'key': 'value'}", treeMaker::expression);
    assertThat(tree.firstToken().value()).isEqualTo("{");
    assertThat(tree.lastToken().value()).isEqualTo("}");
    assertThat(tree.getKind()).isEqualTo(Tree.Kind.DICTIONARY_LITERAL);
    assertThat(tree.elements()).hasSize(1);
    KeyValuePair keyValuePair = tree.elements().iterator().next();
    assertThat(keyValuePair.getKind()).isEqualTo(Tree.Kind.KEY_VALUE_PAIR);
    assertThat(keyValuePair.key().getKind()).isEqualTo(Tree.Kind.STRING_LITERAL);
    assertThat(keyValuePair.colon().value()).isEqualTo(":");
    assertThat(keyValuePair.value().getKind()).isEqualTo(Tree.Kind.STRING_LITERAL);
    assertThat(tree.children()).hasSize(1);

    tree = (DictionaryLiteral) parse("{'key': 'value', 'key2': 'value2'}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);

    tree = (DictionaryLiteral) parse("{** var}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(1);
    keyValuePair = tree.elements().iterator().next();
    assertThat(keyValuePair.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(keyValuePair.starStarToken().value()).isEqualTo("**");

    tree = (DictionaryLiteral) parse("{** var, key: value}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);
  }

  @Test
  public void dict_comprehension() {
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
    assertThat(comprehension.children()).hasSize(4);
    assertThat(comprehension.firstToken().value()).isEqualTo("{");
    assertThat(comprehension.lastToken().value()).isEqualTo("}");
  }

  @Test
  public void set_literal() {
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
    assertThat(tree.children()).hasSize(1);

    tree = (SetLiteral) parse("{ x, y }", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);

    tree = (SetLiteral) parse("{ *x }", treeMaker::expression);
    assertThat(tree.elements()).hasSize(1);
    element = tree.elements().iterator().next();
    assertThat(element.getKind()).isEqualTo(Tree.Kind.STARRED_EXPR);
  }

  @Test
  public void set_comprehension() {
    setRootRule(PythonGrammar.TEST);
    ComprehensionExpression comprehension =
      (ComprehensionExpression) parse("{x-1 for x in [42, 43]}", treeMaker::expression);
    assertThat(comprehension.firstToken().value()).isEqualTo("{");
    assertThat(comprehension.lastToken().value()).isEqualTo("}");
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.SET_COMPREHENSION);
    assertThat(comprehension.resultExpression().getKind()).isEqualTo(Tree.Kind.MINUS);
    assertThat(comprehension.children()).hasSize(2);
    assertThat(comprehension.firstToken().value()).isEqualTo("{");
    assertThat(comprehension.lastToken().value()).isEqualTo("}");
  }

  @Test
  public void repr_expression() {
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
    assertThat(reprExpressionTree.expressionList().expressions().get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(reprExpressionTree.expressionList().expressions().get(1).getKind()).isEqualTo(Tree.Kind.NAME);
  }

  @Test
  public void ellipsis_expression() {
    setRootRule(PythonGrammar.ATOM);
    EllipsisExpression ellipsisExpressionTree = (EllipsisExpression) parse("...", treeMaker::expression);
    assertThat(ellipsisExpressionTree.getKind()).isEqualTo(Tree.Kind.ELLIPSIS);
    assertThat(ellipsisExpressionTree.ellipsis()).extracting(Token::value).containsExactly(".", ".", ".");
    assertThat(ellipsisExpressionTree.children()).hasSize(3);
  }

  @Test
  public void none_expression() {
    setRootRule(PythonGrammar.ATOM);
    NoneExpression noneExpressionTree = (NoneExpression) parse("None", treeMaker::expression);
    assertThat(noneExpressionTree.getKind()).isEqualTo(Tree.Kind.NONE);
    assertThat(noneExpressionTree.none().value()).isEqualTo("None");
    assertThat(noneExpressionTree.children()).hasSize(1);
  }

  @Test
  public void variables() {
    setRootRule(PythonGrammar.ATOM);
    Name name = (Name) parse("foo", treeMaker::expression);
    assertThat(name.isVariable()).isTrue();

    setRootRule(PythonGrammar.ATTRIBUTE_REF);
    QualifiedExpression qualifiedExpressionTree = (QualifiedExpression) parse("a.b", treeMaker::expression);
    assertThat(qualifiedExpressionTree.name().isVariable()).isFalse();

    setRootRule(PythonGrammar.FUNCDEF);
    FunctionDef functionDefTree = parse("def func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.name().isVariable()).isFalse();
  }

  @Test
  public void statements_separators() {
    FileInput tree = parse("foo(); bar()\ntoto()", treeMaker::fileInput);
    List<Statement> statements = tree.statements().statements();

    List<Tree> statementChildren = statements.get(0).children();
    assertThat(statementChildren.get(statementChildren.size()-1).is(Tree.Kind.TOKEN)).isTrue();
    Token token = (Token) statementChildren.get(statementChildren.size()-1);
    assertThat(token.token().getType()).isEqualTo(PythonPunctuator.SEMICOLON);

    statementChildren = statements.get(1).children();
    assertThat(statementChildren.get(statementChildren.size()-1).is(Tree.Kind.TOKEN)).isTrue();
    token = (Token) statementChildren.get(statementChildren.size()-1);
    assertThat(token.token().getType()).isEqualTo(PythonTokenType.NEWLINE);

    tree = parse("foo()\ntoto()", treeMaker::fileInput);
    statements = tree.statements().statements();
    statementChildren = statements.get(0).children();
    assertThat(statementChildren.get(statementChildren.size()-1).is(Tree.Kind.TOKEN)).isTrue();
    token = (Token) statementChildren.get(statementChildren.size()-1);
    assertThat(token.token().getType()).isEqualTo(PythonTokenType.NEWLINE);

    // Check that the second semicolon should be ignored
    tree = parse("foo(); bar();\ntoto()", treeMaker::fileInput);
    statements = tree.statements().statements();
    statementChildren = statements.get(0).children();
    assertThat(statementChildren.get(statementChildren.size()-1).is(Tree.Kind.TOKEN)).isTrue();
    token = (Token) statementChildren.get(statementChildren.size()-1);
    assertThat(token.token().getType()).isEqualTo(PythonPunctuator.SEMICOLON);

    statementChildren = statements.get(1).children();
    assertThat(statementChildren.get(statementChildren.size()-1).is(Tree.Kind.TOKEN)).isTrue();
    token = (Token) statementChildren.get(statementChildren.size()-1);
    assertThat(token.token().getType()).isEqualTo(PythonTokenType.NEWLINE);
  }

  private void assertUnaryExpression(String operator, Tree.Kind kind) {
    setRootRule(PythonGrammar.EXPR);
    Expression parse = parse(operator+"1", treeMaker::expression);
    assertThat(parse.is(kind)).isTrue();
    UnaryExpression unary = (UnaryExpression) parse;
    assertThat(unary.expression().is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(unary.operator().value()).isEqualTo(operator);
    assertThat(unary.children()).hasSize(2);
  }

  private <T extends Tree> T parse(String code, Function<AstNode, T> func) {
    T tree = func.apply(p.parse(code));
    // ensure every visit method of base tree visitor is called without errors
    BaseTreeVisitor visitor = new BaseTreeVisitor();
    tree.accept(visitor);
    return tree;
  }
}
