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
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyAnnotatedAssignmentTree;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyAssertStatementTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyAwaitExpressionTree;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyBreakStatementTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyCompoundAssignmentStatementTree;
import org.sonar.python.api.tree.PyComprehensionExpressionTree;
import org.sonar.python.api.tree.PyComprehensionForTree;
import org.sonar.python.api.tree.PyComprehensionIfTree;
import org.sonar.python.api.tree.PyConditionalExpressionTree;
import org.sonar.python.api.tree.PyContinueStatementTree;
import org.sonar.python.api.tree.PyDecoratorTree;
import org.sonar.python.api.tree.PyDelStatementTree;
import org.sonar.python.api.tree.PyDictCompExpressionTree;
import org.sonar.python.api.tree.PyDictionaryLiteralTree;
import org.sonar.python.api.tree.PyEllipsisExpressionTree;
import org.sonar.python.api.tree.PyElseStatementTree;
import org.sonar.python.api.tree.PyExceptClauseTree;
import org.sonar.python.api.tree.PyExecStatementTree;
import org.sonar.python.api.tree.PyExpressionListTree;
import org.sonar.python.api.tree.PyExpressionStatementTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyGlobalStatementTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyImportFromTree;
import org.sonar.python.api.tree.PyImportNameTree;
import org.sonar.python.api.tree.PyImportStatementTree;
import org.sonar.python.api.tree.PyInExpressionTree;
import org.sonar.python.api.tree.PyIsExpressionTree;
import org.sonar.python.api.tree.PyKeyValuePairTree;
import org.sonar.python.api.tree.PyLambdaExpressionTree;
import org.sonar.python.api.tree.PyListLiteralTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyNoneExpressionTree;
import org.sonar.python.api.tree.PyNonlocalStatementTree;
import org.sonar.python.api.tree.PyNumericLiteralTree;
import org.sonar.python.api.tree.PyParameterTree;
import org.sonar.python.api.tree.PyParenthesizedExpressionTree;
import org.sonar.python.api.tree.PyPassStatementTree;
import org.sonar.python.api.tree.PyPrintStatementTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.PyRaiseStatementTree;
import org.sonar.python.api.tree.PyReprExpressionTree;
import org.sonar.python.api.tree.PyReturnStatementTree;
import org.sonar.python.api.tree.PySetLiteralTree;
import org.sonar.python.api.tree.PySliceExpressionTree;
import org.sonar.python.api.tree.PySliceItemTree;
import org.sonar.python.api.tree.PyStarredExpressionTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.PyStringElementTree;
import org.sonar.python.api.tree.PyStringLiteralTree;
import org.sonar.python.api.tree.PySubscriptionExpressionTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.PyTryStatementTree;
import org.sonar.python.api.tree.PyTupleParameterTree;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.PyTypeAnnotationTree;
import org.sonar.python.api.tree.PyUnaryExpressionTree;
import org.sonar.python.api.tree.PyWhileStatementTree;
import org.sonar.python.api.tree.PyWithItemTree;
import org.sonar.python.api.tree.PyWithStatementTree;
import org.sonar.python.api.tree.PyYieldExpressionTree;
import org.sonar.python.api.tree.PyYieldStatementTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.parser.RuleTest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

public class PythonTreeMakerTest extends RuleTest {

  private final PythonTreeMaker treeMaker = new PythonTreeMaker();

  @Test
  public void fileInputTreeOnEmptyFile() {
    PyFileInputTree pyTree = parse("", treeMaker::fileInput);
    assertThat(pyTree.statements()).isNull();
    assertThat(pyTree.docstring()).isNull();

    pyTree = parse("\"\"\"\n" +
      "This is a module docstring\n" +
      "\"\"\"", treeMaker::fileInput);
    assertThat(pyTree.docstring().value()).isEqualTo("\"\"\"\n" +
      "This is a module docstring\n" +
      "\"\"\"");

    pyTree = parse("if x:\n pass", treeMaker::fileInput);
    PyIfStatementTree ifStmt = (PyIfStatementTree) pyTree.statements().statements().get(0);
    assertThat(ifStmt.body().parent()).isEqualTo(ifStmt);
  }

  @Test
  public void unexpected_statement_should_throw_an_exception() {
    try {
      parse("", treeMaker::statement);
      fail("unexpected ASTNode type for statement should not succeed to be translated to Strongly typed AST");
    } catch (IllegalStateException iae) {
      assertThat(iae).hasMessage("Statement FILE_INPUT not correctly translated to strongly typed AST");
    }
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
    testData.put("pass", PyPassStatementTree.class);
    testData.put("print 'foo'", PyPrintStatementTree.class);
    testData.put("exec foo", PyExecStatementTree.class);
    testData.put("assert foo", PyAssertStatementTree.class);
    testData.put("del foo", PyDelStatementTree.class);
    testData.put("return foo", PyReturnStatementTree.class);
    testData.put("yield foo", PyYieldStatementTree.class);
    testData.put("raise foo", PyRaiseStatementTree.class);
    testData.put("break", PyBreakStatementTree.class);
    testData.put("continue", PyContinueStatementTree.class);
    testData.put("def foo():pass", PyFunctionDefTree.class);
    testData.put("import foo", PyImportStatementTree.class);
    testData.put("from foo import f", PyImportStatementTree.class);
    testData.put("class toto:pass", PyClassDefTree.class);
    testData.put("for foo in bar:pass", PyForStatementTree.class);
    testData.put("async for foo in bar: pass", PyForStatementTree.class);
    testData.put("global foo", PyGlobalStatementTree.class);
    testData.put("nonlocal foo", PyNonlocalStatementTree.class);
    testData.put("while cond: pass", PyWhileStatementTree.class);
    testData.put("'foo'", PyExpressionStatementTree.class);
    testData.put("try: this\nexcept Exception: pass", PyTryStatementTree.class);
    testData.put("with foo, bar as qix : pass", PyWithStatementTree.class);
    testData.put("async with foo, bar as qix : pass", PyWithStatementTree.class);
    testData.put("x = y", PyAssignmentStatementTree.class);
    testData.put("x += y", PyCompoundAssignmentStatementTree.class);

    testData.forEach((c,clazz) -> {
      PyFileInputTree pyTree = parse(c, treeMaker::fileInput);
      PyStatementListTree statementList = pyTree.statements();
      assertThat(statementList.statements()).hasSize(1);
      PyStatementTree stmt = statementList.statements().get(0);
      assertThat(stmt.parent()).isEqualTo(statementList);
      assertThat(stmt).as(c).isInstanceOf(clazz);
    });
  }

  @Test
  public void IfStatement() {
    setRootRule(PythonGrammar.IF_STMT);
    PyIfStatementTree pyIfStatementTree = parse("if x: pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().value()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.body().statements()).hasSize(1);
    assertThat(pyIfStatementTree.children()).hasSize(3);


    pyIfStatementTree = parse("if x: pass\nelse: pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().value()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    PyElseStatementTree elseBranch = pyIfStatementTree.elseBranch();
    assertThat(elseBranch.firstToken().value()).isEqualTo("else");
    assertThat(elseBranch.lastToken().value()).isEqualTo("pass");
    assertThat(elseBranch).isNotNull();
    assertThat(elseBranch.elseKeyword().value()).isEqualTo("else");
    assertThat(elseBranch.body().statements()).hasSize(1);
    assertThat(pyIfStatementTree.children()).hasSize(4);


    pyIfStatementTree = parse("if x: pass\nelif y: pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().value()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.elifBranches()).hasSize(1);
    PyIfStatementTree elif = pyIfStatementTree.elifBranches().get(0);
    assertThat(elif.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(elif.firstToken().value()).isEqualTo("elif");
    assertThat(elif.lastToken().value()).isEqualTo("pass");
    assertThat(elif.isElif()).isTrue();
    assertThat(elif.elseBranch()).isNull();
    assertThat(elif.elifBranches()).isEmpty();
    assertThat(elif.body().statements()).hasSize(1);
    assertThat(pyIfStatementTree.children()).hasSize(4);

    pyIfStatementTree = parse("if x:\n pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().value()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    assertThat(pyIfStatementTree.body().statements()).hasSize(1);

    pyIfStatementTree = parse("if x:\n pass\n pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.body().statements()).hasSize(2);

    // tokens
    AstNode parseTree = p.parse("if x: pass");
    PyIfStatementTree pyFileInputTree = treeMaker.ifStatement(parseTree);
    assertThat(pyFileInputTree.body().tokens().stream().map(PyToken::token)).isEqualTo(parseTree.getFirstChild(PythonGrammar.SUITE).getTokens());
  }

  @Test
  public void printStatement() {
    setRootRule(PythonGrammar.PRINT_STMT);
    AstNode astNode = p.parse("print 'foo'");
    PyPrintStatementTree printStmt = treeMaker.printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().value()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(1);
    assertThat(printStmt.children()).hasSize(2);

    astNode = p.parse("print 'foo', 'bar'");
    printStmt = treeMaker.printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().value()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(2);
    assertThat(printStmt.children()).hasSize(3);

    astNode = p.parse("print >> 'foo'");
    printStmt = treeMaker.printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().value()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(1);
  }

  @Test
  public void execStatement() {
    setRootRule(PythonGrammar.EXEC_STMT);
    AstNode astNode = p.parse("exec 'foo'");
    PyExecStatementTree execStatement = treeMaker.execStatement(astNode);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().value()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNull();
    assertThat(execStatement.localsExpression()).isNull();
    assertThat(execStatement.children()).hasSize(2);

    astNode = p.parse("exec 'foo' in globals");
    execStatement = treeMaker.execStatement(astNode);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().value()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNotNull();
    assertThat(execStatement.localsExpression()).isNull();
    assertThat(execStatement.children()).hasSize(3);

    astNode = p.parse("exec 'foo' in globals, locals");
    execStatement = treeMaker.execStatement(astNode);
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
    PyAssertStatementTree assertStatement = treeMaker.assertStatement(astNode);
    assertThat(assertStatement).isNotNull();
    assertThat(assertStatement.assertKeyword().value()).isEqualTo("assert");
    assertThat(assertStatement.condition()).isNotNull();
    assertThat(assertStatement.message()).isNull();
    assertThat(assertStatement.children()).hasSize(2);

    astNode = p.parse("assert x, y");
    assertStatement = treeMaker.assertStatement(astNode);
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
    PyPassStatementTree passStatement = treeMaker.passStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.passKeyword().value()).isEqualTo("pass");
    assertThat(passStatement.children()).hasSize(1);
  }

  @Test
  public void delStatement() {
    setRootRule(PythonGrammar.DEL_STMT);
    AstNode astNode = p.parse("del foo");
    PyDelStatementTree passStatement = treeMaker.delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().value()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);
    assertThat(passStatement.children()).hasSize(2);


    astNode = p.parse("del foo, bar");
    passStatement = treeMaker.delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().value()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(2);
    assertThat(passStatement.children()).hasSize(3);

    astNode = p.parse("del *foo");
    passStatement = treeMaker.delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().value()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);
    assertThat(passStatement.children()).hasSize(2);
  }

  @Test
  public void returnStatement() {
    setRootRule(PythonGrammar.RETURN_STMT);
    AstNode astNode = p.parse("return foo");
    PyReturnStatementTree returnStatement = treeMaker.returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().value()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(1);
    assertThat(returnStatement.children()).hasSize(2);

    astNode = p.parse("return foo, bar");
    returnStatement = treeMaker.returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().value()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(2);
    assertThat(returnStatement.children()).hasSize(3);

    astNode = p.parse("return");
    returnStatement = treeMaker.returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().value()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(0);
    assertThat(returnStatement.children()).hasSize(1);
  }

  @Test
  public void yieldStatement() {
    setRootRule(PythonGrammar.YIELD_STMT);
    AstNode astNode = p.parse("yield foo");
    PyYieldStatementTree yieldStatement = treeMaker.yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    assertThat(yieldStatement.children()).hasSize(1);
    PyYieldExpressionTree yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.expressions()).hasSize(1);
    assertThat(yieldExpression.children()).hasSize(2);

    astNode = p.parse("yield foo, bar");
    yieldStatement = treeMaker.yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    assertThat(yieldStatement.children()).hasSize(1);
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.yieldKeyword().value()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword()).isNull();
    assertThat(yieldExpression.expressions()).hasSize(2);
    assertThat(yieldExpression.children()).hasSize(3);

    astNode = p.parse("yield from foo");
    yieldStatement = treeMaker.yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    assertThat(yieldStatement.children()).hasSize(1);
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.yieldKeyword().value()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword().value()).isEqualTo("from");
    assertThat(yieldExpression.expressions()).hasSize(1);
    assertThat(yieldExpression.children()).hasSize(3);

    astNode = p.parse("yield");
    yieldStatement = treeMaker.yieldStatement(astNode);
    assertThat(yieldStatement.children()).hasSize(1);
    assertThat(yieldStatement).isNotNull();
  }

  @Test
  public void raiseStatement() {
    setRootRule(PythonGrammar.RAISE_STMT);
    AstNode astNode = p.parse("raise foo");
    PyRaiseStatementTree raiseStatement = treeMaker.raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().value()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).hasSize(1);
    assertThat(raiseStatement.children()).hasSize(2);

    astNode = p.parse("raise foo, bar");
    raiseStatement = treeMaker.raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().value()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).hasSize(2);
    assertThat(raiseStatement.children()).hasSize(3);

    astNode = p.parse("raise foo from bar");
    raiseStatement = treeMaker.raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().value()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword().value()).isEqualTo("from");
    assertThat(raiseStatement.fromExpression()).isNotNull();
    assertThat(raiseStatement.expressions()).hasSize(1);
    assertThat(raiseStatement.children()).hasSize(4);

    astNode = p.parse("raise");
    raiseStatement = treeMaker.raiseStatement(astNode);
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
    PyBreakStatementTree breakStatement = treeMaker.breakStatement(astNode);
    assertThat(breakStatement).isNotNull();
    assertThat(breakStatement.breakKeyword().value()).isEqualTo("break");
    assertThat(breakStatement.children()).hasSize(1);
  }

  @Test
  public void continueStatement() {
    setRootRule(PythonGrammar.CONTINUE_STMT);
    AstNode astNode = p.parse("continue");
    PyContinueStatementTree continueStatement = treeMaker.continueStatement(astNode);
    assertThat(continueStatement).isNotNull();
    assertThat(continueStatement.continueKeyword().value()).isEqualTo("continue");
    assertThat(continueStatement.children()).hasSize(1);
  }

  @Test
  public void importStatement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    AstNode astNode = p.parse("import foo");
    PyImportNameTree importStatement = (PyImportNameTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.firstToken().value()).isEqualTo("import");
    assertThat(importStatement.lastToken().value()).isEqualTo("foo");
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().value()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    PyAliasedNameTree importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("import foo as f");
    importStatement = (PyImportNameTree) treeMaker.importStatement(astNode);
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
    importStatement = (PyImportNameTree) treeMaker.importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().value()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(2);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importedName1.dottedName().names().get(1).name()).isEqualTo("bar");
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("import foo, bar");
    importStatement = (PyImportNameTree) treeMaker.importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().value()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(2);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    PyAliasedNameTree importedName2 = importStatement.modules().get(1);
    assertThat(importedName2.dottedName().names()).hasSize(1);
    assertThat(importedName2.dottedName().names().get(0).name()).isEqualTo("bar");
    assertThat(importStatement.children()).hasSize(3);
  }

  @Test
  public void importFromStatement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    AstNode astNode = p.parse("from foo import f");
    PyImportFromTree importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
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
    PyAliasedNameTree aliasedNameTree = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree.asKeyword()).isNull();
    assertThat(aliasedNameTree.alias()).isNull();
    assertThat(aliasedNameTree.dottedName().names().get(0).name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(4);

    astNode = p.parse("from .foo import f");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).value()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.children()).hasSize(5);

    astNode = p.parse("from ..foo import f");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(2);
    assertThat(importStatement.dottedPrefixForModule().get(0).value()).isEqualTo(".");
    assertThat(importStatement.dottedPrefixForModule().get(1).value()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.children()).hasSize(6);

    astNode = p.parse("from . import f");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).value()).isEqualTo(".");
    assertThat(importStatement.module()).isNull();
    assertThat(importStatement.children()).hasSize(4);

    astNode = p.parse("from foo import f as g");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.importedNames()).hasSize(1);
    aliasedNameTree = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree.asKeyword().value()).isEqualTo("as");
    assertThat(aliasedNameTree.alias().name()).isEqualTo("g");
    assertThat(aliasedNameTree.dottedName().names().get(0).name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(4);

    astNode = p.parse("from foo import f as g, h");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.importedNames()).hasSize(2);
    PyAliasedNameTree aliasedNameTree1 = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree1.asKeyword().value()).isEqualTo("as");
    assertThat(aliasedNameTree1.alias().name()).isEqualTo("g");
    assertThat(aliasedNameTree1.dottedName().names().get(0).name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(5);

    PyAliasedNameTree aliasedNameTree2 = importStatement.importedNames().get(1);
    assertThat(aliasedNameTree2.asKeyword()).isNull();
    assertThat(aliasedNameTree2.alias()).isNull();
    assertThat(aliasedNameTree2.dottedName().names().get(0).name()).isEqualTo("h");

    astNode = p.parse("from foo import *");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.importedNames()).isEmpty();
    assertThat(importStatement.isWildcardImport()).isTrue();
    assertThat(importStatement.wildcard().value()).isEqualTo("*");
    assertThat(importStatement.children()).hasSize(4);
  }

  @Test
  public void globalStatement() {
    setRootRule(PythonGrammar.GLOBAL_STMT);
    AstNode astNode = p.parse("global foo");
    PyGlobalStatementTree globalStatement = treeMaker.globalStatement(astNode);
    assertThat(globalStatement.globalKeyword().value()).isEqualTo("global");
    assertThat(globalStatement.variables()).hasSize(1);
    assertThat(globalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(globalStatement.children()).hasSize(2);

    astNode = p.parse("global foo, bar");
    globalStatement = treeMaker.globalStatement(astNode);
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
    PyNonlocalStatementTree nonlocalStatement = treeMaker.nonlocalStatement(astNode);
    assertThat(nonlocalStatement.nonlocalKeyword().value()).isEqualTo("nonlocal");
    assertThat(nonlocalStatement.variables()).hasSize(1);
    assertThat(nonlocalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(nonlocalStatement.children()).hasSize(2);

    astNode = p.parse("nonlocal foo, bar");
    nonlocalStatement = treeMaker.nonlocalStatement(astNode);
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
    PyFunctionDefTree functionDefTree = treeMaker.funcDefStatement(astNode);
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
    PyTypeAnnotationTree returnType = functionDefTree.returnTypeAnnotation();
    assertThat(returnType.getKind()).isEqualTo(Tree.Kind.RETURN_TYPE_ANNOTATION);
    assertThat(returnType.dash().value()).isEqualTo("-");
    assertThat(returnType.gt().value()).isEqualTo(">");
    assertThat(returnType.expression().getKind()).isEqualTo(Tree.Kind.NAME);

    functionDefTree = parse("@foo\ndef func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.decorators()).hasSize(1);
    PyDecoratorTree decoratorTree = functionDefTree.decorators().get(0);
    assertThat(decoratorTree.getKind()).isEqualTo(Tree.Kind.DECORATOR);
    assertThat(decoratorTree.atToken().value()).isEqualTo("@");
    assertThat(decoratorTree.name().names().get(0).name()).isEqualTo("foo");
    assertThat(decoratorTree.leftPar()).isNull();
    assertThat(decoratorTree.arguments()).isNull();
    assertThat(decoratorTree.rightPar()).isNull();

    functionDefTree = parse("@foo()\n@bar(1)\ndef func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.decorators()).hasSize(2);
    PyDecoratorTree decoratorTree1 = functionDefTree.decorators().get(0);
    assertThat(decoratorTree1.leftPar().value()).isEqualTo("(");
    assertThat(decoratorTree1.arguments()).isNull();
    assertThat(decoratorTree1.rightPar().value()).isEqualTo(")");
    PyDecoratorTree decoratorTree2 = functionDefTree.decorators().get(1);
    assertThat(decoratorTree2.arguments().arguments()).extracting(arg -> arg.expression().getKind()).containsExactly(Tree.Kind.NUMERIC_LITERAL);

    functionDefTree = parse("def func(x, y): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.parameters().all()).hasSize(2);

    functionDefTree = parse("def func(x = 'foo', y): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.parameters().all()).isEqualTo(functionDefTree.parameters().nonTuple());
    List<PyParameterTree> parameters = functionDefTree.parameters().nonTuple();
    assertThat(parameters).hasSize(2);
    PyParameterTree parameter1 = parameters.get(0);
    assertThat(parameter1.name().name()).isEqualTo("x");
    assertThat(parameter1.equalToken().value()).isEqualTo("=");
    assertThat(parameter1.defaultValue().is(Tree.Kind.STRING_LITERAL)).isTrue();
    PyParameterTree parameter2 = parameters.get(1);
    assertThat(parameter2.equalToken()).isNull();
    assertThat(parameter2.defaultValue()).isNull();

    functionDefTree = parse("def func(p1, *p2, **p3): pass", treeMaker::funcDefStatement);
    parameters = functionDefTree.parameters().nonTuple();
    assertThat(parameters).extracting(p -> p.name().name()).containsExactly("p1", "p2", "p3");
    assertThat(parameters).extracting(p -> p.starToken() == null ? null : p.starToken().value()).containsExactly(null, "*", "**");

    functionDefTree = parse("def func(x : int, y): pass", treeMaker::funcDefStatement);
    parameters = functionDefTree.parameters().nonTuple();
    assertThat(parameters).hasSize(2);
    PyTypeAnnotationTree annotation = parameters.get(0).typeAnnotation();
    assertThat(annotation.getKind()).isEqualTo(Tree.Kind.TYPE_ANNOTATION);
    assertThat(annotation.colonToken().value()).isEqualTo(":");
    assertThat(((PyNameTree) annotation.expression()).name()).isEqualTo("int");
    assertThat(annotation.children()).hasSize(2);
    assertThat(parameters.get(1).typeAnnotation()).isNull();

    functionDefTree = parse("def func(a, ((b, c), d)): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.parameters().all()).hasSize(2);
    assertThat(functionDefTree.parameters().all()).extracting(Tree::getKind).containsExactly(Tree.Kind.PARAMETER, Tree.Kind.TUPLE_PARAMETER);
    PyTupleParameterTree tupleParam = (PyTupleParameterTree) functionDefTree.parameters().all().get(1);
    assertThat(tupleParam.openingParenthesis().value()).isEqualTo("(");
    assertThat(tupleParam.closingParenthesis().value()).isEqualTo(")");
    assertThat(tupleParam.parameters()).extracting(Tree::getKind).containsExactly(Tree.Kind.TUPLE_PARAMETER, Tree.Kind.PARAMETER);
    assertThat(tupleParam.commas()).extracting(PyToken::value).containsExactly(",");
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
    PyClassDefTree classDefTree = treeMaker.classDefStatement(astNode);
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
    PyDecoratorTree decorator = classDefTree.decorators().get(0);
    assertThat(decorator.name().names()).extracting(PyNameTree::name).containsExactly("foo", "bar");

    astNode = p.parse("class clazz:\n  def foo(): pass");
    classDefTree = treeMaker.classDefStatement(astNode);
    PyFunctionDefTree funcDef = (PyFunctionDefTree) classDefTree.body().statements().get(0);
    assertThat(funcDef.isMethodDefinition()).isTrue();

    astNode = p.parse("class ClassWithDocstring:\n" +
      "\t\"\"\"This is a docstring\"\"\"\n" +
      "\tpass");
    classDefTree = treeMaker.classDefStatement(astNode);
    assertThat(classDefTree.docstring().value()).isEqualTo("\"\"\"This is a docstring\"\"\"");
    assertThat(classDefTree.children()).hasSize(5);
  }

  @Test
  public void for_statement() {
    setRootRule(PythonGrammar.FOR_STMT);
    AstNode astNode = p.parse("for foo in bar: pass");
    PyForStatementTree pyForStatementTree = treeMaker.forStatement(astNode);
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
    assertThat(pyForStatementTree.children()).hasSize(9);

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
    PyWhileStatementTreeImpl whileStatement = treeMaker.whileStatement(astNode);
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
    assertThat(whileStatement.children()).hasSize(7);

    assertThat(whileStatement.whileKeyword().value()).isEqualTo("while");
    assertThat(whileStatement.colon().value()).isEqualTo(":");
    assertThat(whileStatement.elseKeyword().value()).isEqualTo("else");
    assertThat(whileStatement.elseColon().value()).isEqualTo(":");

  }

  @Test
  public void expression_statement() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("'foo'");
    PyExpressionStatementTree expressionStatement = treeMaker.expressionStatement(astNode);
    assertThat(expressionStatement.expressions()).hasSize(1);
    assertThat(expressionStatement.children()).hasSize(1);

    astNode = p.parse("'foo', 'bar'");
    expressionStatement = treeMaker.expressionStatement(astNode);
    assertThat(expressionStatement.expressions()).hasSize(2);
    assertThat(expressionStatement.children()).hasSize(2);
  }

  @Test
  public void assignement_statement() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x = y");
    PyAssignmentStatementTree pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.firstToken().value()).isEqualTo("x");
    assertThat(pyAssignmentStatement.lastToken().value()).isEqualTo("y");
    PyNameTree assigned = (PyNameTree) pyAssignmentStatement.assignedValue();
    PyNameTree lhs = (PyNameTree) pyAssignmentStatement.lhsExpressions().get(0).expressions().get(0);
    assertThat(assigned.name()).isEqualTo("y");
    assertThat(lhs.name()).isEqualTo("x");
    assertThat(pyAssignmentStatement.children()).hasSize(3);

    astNode = p.parse("x = y = z");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.equalTokens()).hasSize(2);
    assertThat(pyAssignmentStatement.children()).hasSize(5);
    assigned = (PyNameTree) pyAssignmentStatement.assignedValue();
    lhs = (PyNameTree) pyAssignmentStatement.lhsExpressions().get(0).expressions().get(0);
    PyNameTree lhs2 = (PyNameTree) pyAssignmentStatement.lhsExpressions().get(1).expressions().get(0);
    assertThat(assigned.name()).isEqualTo("z");
    assertThat(lhs.name()).isEqualTo("x");
    assertThat(lhs2.name()).isEqualTo("y");

    astNode = p.parse("a,b = x");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.children()).hasSize(3);
    assigned = (PyNameTree) pyAssignmentStatement.assignedValue();
    List<PyExpressionTree> expressions = pyAssignmentStatement.lhsExpressions().get(0).expressions();
    assertThat(assigned.name()).isEqualTo("x");
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(expressions.get(1).getKind()).isEqualTo(Tree.Kind.NAME);

    astNode = p.parse("x = a,b");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.children()).hasSize(3);
    expressions = pyAssignmentStatement.lhsExpressions().get(0).expressions();
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyAssignmentStatement.assignedValue().getKind()).isEqualTo(Tree.Kind.TUPLE);

    astNode = p.parse("x = yield 1");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.children()).hasSize(3);
    expressions = pyAssignmentStatement.lhsExpressions().get(0).expressions();
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyAssignmentStatement.assignedValue().getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);

    // FIXME: lhs expression list shouldn't allow yield expressions. We need to change the grammar
    astNode = p.parse("x = yield 1 = y");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.children()).hasSize(5);
    List<PyExpressionListTree> lhsExpressions = pyAssignmentStatement.lhsExpressions();
    assertThat(lhsExpressions.get(1).expressions().get(0).getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);
    assertThat(pyAssignmentStatement.assignedValue().getKind()).isEqualTo(Tree.Kind.NAME);
  }

  @Test
  public void annotated_assignment() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x : string = 1");
    PyAnnotatedAssignmentTree annAssign = treeMaker.annotatedAssignment(astNode);
    assertThat(annAssign.firstToken().value()).isEqualTo("x");
    assertThat(annAssign.lastToken().value()).isEqualTo("1");
    assertThat(annAssign.getKind()).isEqualTo(Tree.Kind.ANNOTATED_ASSIGNMENT);
    assertThat(annAssign.children()).hasSize(5);
    assertThat(annAssign.variable().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((PyNameTree) annAssign.variable()).name()).isEqualTo("x");
    assertThat(annAssign.assignedValue().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(annAssign.equalToken().value()).isEqualTo("=");
    assertThat(annAssign.annotation().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((PyNameTree) annAssign.annotation()).name()).isEqualTo("string");
    assertThat(annAssign.colonToken().value()).isEqualTo(":");

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    astNode = p.parse("x : string");
    annAssign = treeMaker.annotatedAssignment(astNode);
    assertThat(annAssign.variable().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((PyNameTree) annAssign.variable()).name()).isEqualTo("x");
    assertThat(annAssign.annotation().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((PyNameTree) annAssign.annotation()).name()).isEqualTo("string");
    assertThat(annAssign.assignedValue()).isNull();
    assertThat(annAssign.equalToken()).isNull();
  }

  @Test
  public void compound_assignement_statement() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x += y");
    PyCompoundAssignmentStatementTree pyCompoundAssignmentStatement = treeMaker.compoundAssignment(astNode);
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.children()).hasSize(3);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().value()).isEqualTo("+=");
    assertThat(pyCompoundAssignmentStatement.lhsExpression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyCompoundAssignmentStatement.rhsExpression().getKind()).isEqualTo(Tree.Kind.NAME);

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    astNode = p.parse("x,y,z += 1");
    pyCompoundAssignmentStatement = treeMaker.compoundAssignment(astNode);
    assertThat(pyCompoundAssignmentStatement.firstToken().value()).isEqualTo("x");
    assertThat(pyCompoundAssignmentStatement.lastToken().value()).isEqualTo("1");
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.children()).hasSize(3);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().value()).isEqualTo("+=");
    assertThat(pyCompoundAssignmentStatement.lhsExpression().getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(pyCompoundAssignmentStatement.rhsExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    astNode = p.parse("x += yield y");
    pyCompoundAssignmentStatement = treeMaker.compoundAssignment(astNode);
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.children()).hasSize(3);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().value()).isEqualTo("+=");
    assertThat(pyCompoundAssignmentStatement.lhsExpression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyCompoundAssignmentStatement.rhsExpression().getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);

    astNode = p.parse("x *= z");
    pyCompoundAssignmentStatement = treeMaker.compoundAssignment(astNode);
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().value()).isEqualTo("*=");
  }

  @Test
  public void try_statement() {
    setRootRule(PythonGrammar.TRY_STMT);
    AstNode astNode = p.parse("try: pass\nexcept Error: pass");
    PyTryStatementTree tryStatement = treeMaker.tryStatement(astNode);
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
    assertThat(tryStatement.children()).hasSize(3);


    astNode = p.parse("try: pass\nexcept Error: pass\nexcept Error: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.elseClause()).isNull();
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.exceptClauses()).hasSize(2);
    assertThat(tryStatement.children()).hasSize(4);

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
    assertThat(tryStatement.children()).hasSize(4);

    astNode = p.parse("try: pass\nexcept Error: pass\nelse: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.elseClause().elseKeyword().value()).isEqualTo("else");
    assertThat(tryStatement.elseClause().firstToken().value()).isEqualTo("else");
    assertThat(tryStatement.elseClause().lastToken().value()).isEqualTo("pass");
    assertThat(tryStatement.elseClause().body().statements()).hasSize(1);
    assertThat(tryStatement.children()).hasSize(4);

    astNode = p.parse("try: pass\nexcept Error as e: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    PyExceptClauseTree exceptClause = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause.asKeyword().value()).isEqualTo("as");
    assertThat(exceptClause.commaToken()).isNull();
    assertThat(exceptClause.exceptionInstance()).isNotNull();
    assertThat(tryStatement.children()).hasSize(3);

    astNode = p.parse("try: pass\nexcept Error, e: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().value()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    exceptClause = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause.asKeyword()).isNull();
    assertThat(exceptClause.commaToken().value()).isEqualTo(",");
    assertThat(exceptClause.exceptionInstance()).isNotNull();
    assertThat(tryStatement.children()).hasSize(3);
  }

  @Test
  public void async_statement() {
    setRootRule(PythonGrammar.ASYNC_STMT);
    AstNode astNode = p.parse("async for foo in bar: pass");
    PyForStatementTree pyForStatementTree = new PythonTreeMaker().forStatement(astNode);
    assertThat(pyForStatementTree.isAsync()).isTrue();
    assertThat(pyForStatementTree.asyncKeyword().value()).isEqualTo("async");
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body().statements()).hasSize(1);
    assertThat(pyForStatementTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody()).isNull();
    assertThat(pyForStatementTree.children()).hasSize(8);

    PyWithStatementTree withStatement = parse("async with foo : pass", treeMaker::withStatement);
    assertThat(withStatement.isAsync()).isTrue();
    assertThat(withStatement.asyncKeyword().value()).isEqualTo("async");
    PyWithItemTree pyWithItemTree = withStatement.withItems().get(0);
    assertThat(pyWithItemTree.test()).isNotNull();
    assertThat(pyWithItemTree.as()).isNull();
    assertThat(pyWithItemTree.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(4);
  }

  @Test
  public void with_statement() {
    setRootRule(PythonGrammar.WITH_STMT);
    PyWithStatementTree withStatement = parse("with foo : pass", treeMaker::withStatement);
    assertThat(withStatement.firstToken().value()).isEqualTo("with");
    assertThat(withStatement.lastToken().value()).isEqualTo("pass");
    assertThat(withStatement.isAsync()).isFalse();
    assertThat(withStatement.asyncKeyword()).isNull();
    assertThat(withStatement.withItems()).hasSize(1);
    PyWithItemTree pyWithItemTree = withStatement.withItems().get(0);
    assertThat(pyWithItemTree.test()).isNotNull();
    assertThat(pyWithItemTree.as()).isNull();
    assertThat(pyWithItemTree.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(4);


    withStatement = parse("with foo as bar, qix : pass", treeMaker::withStatement);
    assertThat(withStatement.withItems()).hasSize(2);
    pyWithItemTree = withStatement.withItems().get(0);
    assertThat(pyWithItemTree.firstToken().value()).isEqualTo("foo");
    assertThat(pyWithItemTree.lastToken().value()).isEqualTo("bar");
    assertThat(pyWithItemTree.test()).isNotNull();
    assertThat(pyWithItemTree.as()).isNotNull();
    assertThat(pyWithItemTree.expression()).isNotNull();
    pyWithItemTree = withStatement.withItems().get(1);
    assertThat(pyWithItemTree.test()).isNotNull();
    assertThat(pyWithItemTree.firstToken().value()).isEqualTo("qix");
    assertThat(pyWithItemTree.lastToken().value()).isEqualTo("qix");
    assertThat(pyWithItemTree.as()).isNull();
    assertThat(pyWithItemTree.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(5);
  }

  @Test
  public void verify_expected_expression() {
    Map<String, Class<? extends Tree>> testData = new HashMap<>();
    testData.put("foo", PyNameTree.class);
    testData.put("foo.bar", PyQualifiedExpressionTree.class);
    testData.put("foo()", PyCallExpressionTree.class);
    testData.put("lambda x: x", PyLambdaExpressionTree.class);

    testData.forEach((c,clazz) -> {
      PyFileInputTree pyTree = parse(c, treeMaker::fileInput);
      assertThat(pyTree.statements().statements()).hasSize(1);
      PyExpressionStatementTree expressionStmt = (PyExpressionStatementTree) pyTree.statements().statements().get(0);
      assertThat(expressionStmt).as(c).isInstanceOf(PyExpressionStatementTree.class);
      assertThat(expressionStmt.expressions().get(0)).as(c).isInstanceOf(clazz);
    });
  }

  @Test
  public void call_expression() {
    setRootRule(PythonGrammar.CALL_EXPR);
    PyCallExpressionTree callExpression = parse("foo()", treeMaker::callExpression);
    assertThat(callExpression.argumentList()).isNull();
    assertThat(callExpression.firstToken().value()).isEqualTo("foo");
    assertThat(callExpression.lastToken().value()).isEqualTo(")");
    assertThat(callExpression.arguments()).isEmpty();
    PyNameTree name = (PyNameTree) callExpression.callee();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(3);
    assertThat(callExpression.leftPar().value()).isEqualTo("(");
    assertThat(callExpression.rightPar().value()).isEqualTo(")");

    callExpression = parse("foo(x, y)", treeMaker::callExpression);
    assertThat(callExpression.argumentList().arguments()).hasSize(2);
    assertThat(callExpression.arguments()).hasSize(2);
    PyNameTree firstArg = (PyNameTree) callExpression.argumentList().arguments().get(0).expression();
    PyNameTree sndArg = (PyNameTree) callExpression.argumentList().arguments().get(1).expression();
    assertThat(firstArg.name()).isEqualTo("x");
    assertThat(sndArg.name()).isEqualTo("y");
    name = (PyNameTree) callExpression.callee();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(4);

    callExpression = parse("foo.bar()", treeMaker::callExpression);
    assertThat(callExpression.argumentList()).isNull();
    PyQualifiedExpressionTree callee = (PyQualifiedExpressionTree) callExpression.callee();
    assertThat(callExpression.firstToken().value()).isEqualTo("foo");
    assertThat(callExpression.lastToken().value()).isEqualTo(")");
    assertThat(callee.name().name()).isEqualTo("bar");
    assertThat(((PyNameTree) callee.qualifier()).name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(3);
  }

  @Test
  public void combinations_with_call_expressions() {
    setRootRule(PythonGrammar.TEST);

    PyCallExpressionTree nestingCall = (PyCallExpressionTree) parse("foo('a').bar(42)", treeMaker::expression);
    assertThat(nestingCall.argumentList().arguments()).extracting(t -> t.expression().getKind()).containsExactly(Tree.Kind.NUMERIC_LITERAL);
    PyQualifiedExpressionTree callee = (PyQualifiedExpressionTree) nestingCall.callee();
    assertThat(callee.name().name()).isEqualTo("bar");
    assertThat(callee.qualifier().firstToken().value()).isEqualTo("foo");
    assertThat(callee.qualifier().lastToken().value()).isEqualTo(")");
    assertThat(callee.qualifier().getKind()).isEqualTo(Tree.Kind.CALL_EXPR);

    nestingCall = (PyCallExpressionTree) parse("foo('a').bar()", treeMaker::expression);
    assertThat(nestingCall.argumentList()).isNull();

    PyCallExpressionTree callOnSubscription = (PyCallExpressionTree) parse("a[42](arg)", treeMaker::expression);
    PySubscriptionExpressionTree subscription = (PySubscriptionExpressionTree) callOnSubscription.callee();
    assertThat(((PyNameTree) subscription.object()).name()).isEqualTo("a");
    assertThat(subscription.subscripts().expressions()).extracting(Tree::getKind).containsExactly(Tree.Kind.NUMERIC_LITERAL);
    assertThat(((PyNameTree) callOnSubscription.argumentList().arguments().get(0).expression()).name()).isEqualTo("arg");
  }

  @Test
  public void attributeRef_expression() {
    setRootRule(PythonGrammar.ATTRIBUTE_REF);
    PyQualifiedExpressionTree qualifiedExpression = parse("foo.bar", treeMaker::qualifiedExpression);
    assertThat(qualifiedExpression.name().name()).isEqualTo("bar");
    PyExpressionTree qualifier = qualifiedExpression.qualifier();
    assertThat(qualifier).isInstanceOf(PyNameTree.class);
    assertThat(((PyNameTree) qualifier).name()).isEqualTo("foo");
    assertThat(qualifiedExpression.children()).hasSize(3);

    qualifiedExpression = parse("foo.bar.baz", treeMaker::qualifiedExpression);
    assertThat(qualifiedExpression.name().name()).isEqualTo("baz");
    assertThat(qualifiedExpression.firstToken().value()).isEqualTo("foo");
    assertThat(qualifiedExpression.lastToken().value()).isEqualTo("baz");
    assertThat(qualifiedExpression.qualifier()).isInstanceOf(PyQualifiedExpressionTree.class);
    PyQualifiedExpressionTree qualExpr = (PyQualifiedExpressionTree) qualifiedExpression.qualifier();
    assertThat(qualExpr.name().name()).isEqualTo("bar");
    assertThat(qualExpr.firstToken().value()).isEqualTo("foo");
    assertThat(qualExpr.lastToken().value()).isEqualTo("bar");
    assertThat(qualExpr.qualifier()).isInstanceOf(PyNameTree.class);
    PyNameTree name = (PyNameTree) qualExpr.qualifier();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(qualifiedExpression.children()).hasSize(3);
  }

  @Test
  public void argument() {
    setRootRule(PythonGrammar.ARGUMENT);
    PyArgumentTree argumentTree = parse("foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNull();
    assertThat(argumentTree.keywordArgument()).isNull();
    PyNameTree name = (PyNameTree) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNull();
    assertThat(argumentTree.starStarToken()).isNull();
    assertThat(argumentTree.children()).hasSize(1);

    argumentTree = parse("*foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNull();
    assertThat(argumentTree.keywordArgument()).isNull();
    name = (PyNameTree) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNotNull();
    assertThat(argumentTree.starStarToken()).isNull();
    assertThat(argumentTree.children()).hasSize(2);

    argumentTree = parse("**foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNull();
    assertThat(argumentTree.keywordArgument()).isNull();
    name = (PyNameTree) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNull();
    assertThat(argumentTree.starStarToken()).isNotNull();
    assertThat(argumentTree.children()).hasSize(2);

    argumentTree = parse("bar=foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNotNull();
    PyNameTree keywordArgument = argumentTree.keywordArgument();
    assertThat(keywordArgument.name()).isEqualTo("bar");
    name = (PyNameTree) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNull();
    assertThat(argumentTree.starStarToken()).isNull();
    assertThat(argumentTree.children()).hasSize(3);
  }

  @Test
  public void binary_expressions() {
    setRootRule(PythonGrammar.TEST);

    PyBinaryExpressionTree simplePlus = binaryExpression("a + b");
    assertThat(simplePlus.leftOperand()).isInstanceOf(PyNameTree.class);
    assertThat(simplePlus.operator().value()).isEqualTo("+");
    assertThat(simplePlus.rightOperand()).isInstanceOf(PyNameTree.class);
    assertThat(simplePlus.getKind()).isEqualTo(Tree.Kind.PLUS);
    assertThat(simplePlus.children()).hasSize(3);

    PyBinaryExpressionTree compoundPlus = binaryExpression("a + b - c");
    assertThat(compoundPlus.leftOperand()).isInstanceOf(PyBinaryExpressionTree.class);
    assertThat(compoundPlus.children()).hasSize(3);
    assertThat(compoundPlus.operator().value()).isEqualTo("-");
    assertThat(compoundPlus.rightOperand()).isInstanceOf(PyNameTree.class);
    assertThat(compoundPlus.getKind()).isEqualTo(Tree.Kind.MINUS);
    PyBinaryExpressionTree compoundPlusLeft = (PyBinaryExpressionTree) compoundPlus.leftOperand();
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

  private PyBinaryExpressionTree binaryExpression(String code) {
    PyExpressionTree exp = parse(code, treeMaker::expression);
    assertThat(exp).isInstanceOf(PyBinaryExpressionTree.class);
    return (PyBinaryExpressionTree) exp;
  }

  @Test
  public void in_expressions() {
    setRootRule(PythonGrammar.TEST);

    PyInExpressionTree in = (PyInExpressionTree) binaryExpression("1 in [a]");
    assertThat(in.getKind()).isEqualTo(Tree.Kind.IN);
    assertThat(in.operator().value()).isEqualTo("in");
    assertThat(in.leftOperand().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(in.rightOperand().getKind()).isEqualTo(Tree.Kind.LIST_LITERAL);
    assertThat(in.notToken()).isNull();

    PyInExpressionTree notIn = (PyInExpressionTree) binaryExpression("1 not in [a]");
    assertThat(notIn.getKind()).isEqualTo(Tree.Kind.IN);
    assertThat(notIn.operator().value()).isEqualTo("in");
    assertThat(notIn.notToken()).isNotNull();
  }

  @Test
  public void is_expressions() {
    setRootRule(PythonGrammar.TEST);

    PyIsExpressionTree in = (PyIsExpressionTree) binaryExpression("a is 1");
    assertThat(in.getKind()).isEqualTo(Tree.Kind.IS);
    assertThat(in.operator().value()).isEqualTo("is");
    assertThat(in.leftOperand().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(in.rightOperand().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(in.notToken()).isNull();

    PyIsExpressionTree notIn = (PyIsExpressionTree) binaryExpression("a is not 1");
    assertThat(notIn.getKind()).isEqualTo(Tree.Kind.IS);
    assertThat(notIn.operator().value()).isEqualTo("is");
    assertThat(notIn.notToken()).isNotNull();
  }

  @Test
  public void starred_expression() {
    setRootRule(PythonGrammar.STAR_EXPR);
    PyStarredExpressionTree starred = (PyStarredExpressionTree) parse("*a", treeMaker::expression);
    assertThat(starred.getKind()).isEqualTo(Tree.Kind.STARRED_EXPR);
    assertThat(starred.starToken().value()).isEqualTo("*");
    assertThat(starred.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(starred.children()).hasSize(2);
  }

  @Test
  public void await_expression() {
    setRootRule(PythonGrammar.TEST);
    PyAwaitExpressionTree expr = (PyAwaitExpressionTree) parse("await x", treeMaker::expression);
    assertThat(expr.getKind()).isEqualTo(Tree.Kind.AWAIT);
    assertThat(expr.awaitToken().value()).isEqualTo("await");
    assertThat(expr.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(expr.children()).hasSize(2);

    PyBinaryExpressionTree awaitWithPower = binaryExpression("await a ** 3");
    assertThat(awaitWithPower.getKind()).isEqualTo(Tree.Kind.POWER);
    assertThat(awaitWithPower.leftOperand().getKind()).isEqualTo(Tree.Kind.AWAIT);
    assertThat(awaitWithPower.rightOperand().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
  }

  @Test
  public void subscription_expressions() {
    setRootRule(PythonGrammar.TEST);

    PySubscriptionExpressionTree expr = (PySubscriptionExpressionTree) parse("x[a]", treeMaker::expression);
    assertThat(expr.getKind()).isEqualTo(Tree.Kind.SUBSCRIPTION);
    assertThat(((PyNameTree) expr.object()).name()).isEqualTo("x");
    assertThat(((PyNameTree) expr.subscripts().expressions().get(0)).name()).isEqualTo("a");
    assertThat(expr.leftBracket().value()).isEqualTo("[");
    assertThat(expr.rightBracket().value()).isEqualTo("]");
    assertThat(expr.children()).hasSize(4);

    PySubscriptionExpressionTree multipleSubscripts = (PySubscriptionExpressionTree) parse("x[a, 42]", treeMaker::expression);
    assertThat(multipleSubscripts.subscripts().expressions()).extracting(Tree::getKind)
      .containsExactly(Tree.Kind.NAME, Tree.Kind.NUMERIC_LITERAL);
  }

  @Test
  public void slice_expressions() {
    setRootRule(PythonGrammar.TEST);

    PySliceExpressionTree expr = (PySliceExpressionTree) parse("x[a:b:c]", treeMaker::expression);
    assertThat(expr.getKind()).isEqualTo(Tree.Kind.SLICE_EXPR);
    assertThat(expr.object().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(expr.leftBracket().value()).isEqualTo("[");
    assertThat(expr.rightBracket().value()).isEqualTo("]");
    assertThat(expr.children()).hasSize(4);
    assertThat(expr.sliceList().getKind()).isEqualTo(Tree.Kind.SLICE_LIST);
    assertThat(expr.sliceList().children()).hasSize(1);
    assertThat(expr.sliceList().slices().get(0).getKind()).isEqualTo(Tree.Kind.SLICE_ITEM);

    PySliceExpressionTree multipleSlices = (PySliceExpressionTree) parse("x[a, b:c, :]", treeMaker::expression);
    List<Tree> slices = multipleSlices.sliceList().slices();
    assertThat(slices).extracting(Tree::getKind).containsExactly(Tree.Kind.NAME, Tree.Kind.SLICE_ITEM, Tree.Kind.SLICE_ITEM);
    assertThat(multipleSlices.sliceList().separators()).extracting(PyToken::value).containsExactly(",", ",");
  }

  @Test
  public void qualified_with_slice() {
    setRootRule(PythonGrammar.TEST);
    PyQualifiedExpressionTree qualifiedWithSlice = (PyQualifiedExpressionTree) parse("x[a:b].foo", treeMaker::expression);
    assertThat(qualifiedWithSlice.qualifier().getKind()).isEqualTo(Tree.Kind.SLICE_EXPR);
  }

  @Test
  public void slice() {
    setRootRule(PythonGrammar.SUBSCRIPT);

    PySliceItemTree slice = parse("a:b:c", treeMaker::sliceItem);
    assertThat(((PyNameTree) slice.lowerBound()).name()).isEqualTo("a");
    assertThat(((PyNameTree) slice.upperBound()).name()).isEqualTo("b");
    assertThat(((PyNameTree) slice.stride()).name()).isEqualTo("c");
    assertThat(slice.boundSeparator().value()).isEqualTo(":");
    assertThat(slice.strideSeparator().value()).isEqualTo(":");
    assertThat(slice.children()).hasSize(5);

    PySliceItemTree trivial = parse(":", treeMaker::sliceItem);
    assertThat(trivial.lowerBound()).isNull();
    assertThat(trivial.upperBound()).isNull();
    assertThat(trivial.stride()).isNull();
    assertThat(trivial.strideSeparator()).isNull();

    PySliceItemTree lowerBoundOnly = parse("a:", treeMaker::sliceItem);
    assertThat(((PyNameTree) lowerBoundOnly.lowerBound()).name()).isEqualTo("a");
    assertThat(lowerBoundOnly.upperBound()).isNull();
    assertThat(lowerBoundOnly.stride()).isNull();
    assertThat(lowerBoundOnly.strideSeparator()).isNull();

    PySliceItemTree upperBoundOnly = parse(":a", treeMaker::sliceItem);
    assertThat(upperBoundOnly.lowerBound()).isNull();
    assertThat(((PyNameTree) upperBoundOnly.upperBound()).name()).isEqualTo("a");
    assertThat(upperBoundOnly.stride()).isNull();
    assertThat(upperBoundOnly.strideSeparator()).isNull();

    PySliceItemTree strideOnly = parse("::a", treeMaker::sliceItem);
    assertThat(strideOnly.lowerBound()).isNull();
    assertThat(strideOnly.upperBound()).isNull();
    assertThat(((PyNameTree) strideOnly.stride()).name()).isEqualTo("a");
    assertThat(strideOnly.strideSeparator()).isNotNull();

    PySliceItemTree strideContainingOnlyColon = parse("::", treeMaker::sliceItem);
    assertThat(strideContainingOnlyColon.lowerBound()).isNull();
    assertThat(strideContainingOnlyColon.upperBound()).isNull();
    assertThat(strideContainingOnlyColon.strideSeparator()).isNotNull();
  }

  @Test
  public void lambda_expr() {
    setRootRule(PythonGrammar.LAMBDEF);
    PyLambdaExpressionTree lambdaExpressionTree = parse("lambda x: x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.expression()).isInstanceOf(PyNameTree.class);
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
    assertThat(((PyTupleParameterTree) lambdaExpressionTree.parameters().all().get(0)).parameters()).hasSize(2);
    assertThat(lambdaExpressionTree.children()).hasSize(4);

    lambdaExpressionTree = parse("lambda *a, **b: 42", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.parameters().nonTuple()).hasSize(2);
    PyParameterTree starArg = lambdaExpressionTree.parameters().nonTuple().get(0);
    assertThat(starArg.starToken().value()).isEqualTo("*");
    assertThat(starArg.name().name()).isEqualTo("a");
    PyParameterTree starStarArg = lambdaExpressionTree.parameters().nonTuple().get(1);
    assertThat(starStarArg.starToken().value()).isEqualTo("**");
    assertThat(starStarArg.name().name()).isEqualTo("b");

    lambdaExpressionTree = parse("lambda x: x if x > 1 else 0", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.getKind()).isEqualTo(Tree.Kind.LAMBDA);
    assertThat(lambdaExpressionTree.expression()).isInstanceOf(PyConditionalExpressionTree.class);

    setRootRule(PythonGrammar.LAMBDEF_NOCOND);
    lambdaExpressionTree = parse("lambda x: x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.getKind()).isEqualTo(Tree.Kind.LAMBDA);
    assertThat(lambdaExpressionTree.expression()).isInstanceOf(PyNameTree.class);
  }

  @Test
  public void numeric_literal_expression() {
    setRootRule(PythonGrammar.ATOM);
    PyExpressionTree parse = parse("12", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    PyNumericLiteralTree numericLiteral = (PyNumericLiteralTree) parse;
    assertThat(numericLiteral.valueAsLong()).isEqualTo(12L);
    assertThat(numericLiteral.valueAsString()).isEqualTo("12");
    assertThat(numericLiteral.children()).isEmpty();

    parse = parse("12L", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    numericLiteral = (PyNumericLiteralTree) parse;
    assertThat(numericLiteral.valueAsLong()).isEqualTo(12L);
    assertThat(numericLiteral.valueAsString()).isEqualTo("12L");
    assertThat(numericLiteral.children()).isEmpty();

    parse = parse("3_0", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    numericLiteral = (PyNumericLiteralTree) parse;
    assertThat(numericLiteral.valueAsLong()).isEqualTo(30L);
    assertThat(numericLiteral.valueAsString()).isEqualTo("3_0");
    assertThat(numericLiteral.children()).isEmpty();

    parse = parse("0b01", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    numericLiteral = (PyNumericLiteralTree) parse;
    assertThat(numericLiteral.valueAsLong()).isEqualTo(1L);
    assertThat(numericLiteral.valueAsString()).isEqualTo("0b01");
    assertThat(numericLiteral.children()).isEmpty();

    parse = parse("0B01", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    numericLiteral = (PyNumericLiteralTree) parse;
    assertThat(numericLiteral.valueAsLong()).isEqualTo(1L);
    assertThat(numericLiteral.valueAsString()).isEqualTo("0B01");
    assertThat(numericLiteral.children()).isEmpty();
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
  }

  private void assertStringLiteral(String fullValue, String trimmedQuoteValue) {
    assertStringLiteral(fullValue, trimmedQuoteValue, "");
  }

  private void assertStringLiteral(String fullValue, String trimmedQuoteValue, String prefix) {
    PyExpressionTree parse = parse(fullValue, treeMaker::expression);
    assertThat(parse.is(Tree.Kind.STRING_LITERAL)).isTrue();
    PyStringLiteralTree stringLiteral = (PyStringLiteralTree) parse;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    PyStringElementTree firstElement = stringLiteral.stringElements().get(0);
    assertThat(firstElement.value()).isEqualTo(fullValue);
    assertThat(firstElement.trimmedQuotesValue()).isEqualTo(trimmedQuoteValue);
    assertThat(firstElement.prefix()).isEqualTo(prefix);
    assertThat(firstElement.children()).isEmpty();
  }

  @Test
  public void multiline_string_literal_expression() {
    setRootRule(PythonGrammar.ATOM);
    PyExpressionTree parse = parse("('Hello \\ ' #Noncompliant\n            'world')", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.PARENTHESIZED)).isTrue();
    PyParenthesizedExpressionTree parenthesized = (PyParenthesizedExpressionTree) parse;
    assertThat(parenthesized.expression().is(Tree.Kind.STRING_LITERAL)).isTrue();
    PyStringLiteralTree pyStringLiteralTree = (PyStringLiteralTree) parenthesized.expression();
    assertThat(pyStringLiteralTree.children()).hasSize(2);
    assertThat(pyStringLiteralTree.stringElements().size()).isEqualTo(2);
    assertThat(pyStringLiteralTree.stringElements().get(0).value()).isEqualTo("\'Hello \\ '");
    PyStringElementTree firstElement = pyStringLiteralTree.stringElements().get(0);
    PyStringElementTree secondElement = pyStringLiteralTree.stringElements().get(1);
    assertThat(secondElement.value()).isEqualTo("'world'");
    assertThat(firstElement.trimmedQuotesValue()).isEqualTo("Hello \\ ");
    assertThat(secondElement.trimmedQuotesValue()).isEqualTo("world");
  }

  @Test
  public void list_literal() {
    setRootRule(PythonGrammar.ATOM);
    PyExpressionTree parse = parse("[1, \"foo\"]", treeMaker::expression);
    assertThat(parse.is(Tree.Kind.LIST_LITERAL)).isTrue();
    assertThat(parse.firstToken().value()).isEqualTo("[");
    assertThat(parse.lastToken().value()).isEqualTo("]");
    PyListLiteralTree listLiteralTree = (PyListLiteralTree) parse;
    List<PyExpressionTree> expressions = listLiteralTree.elements().expressions();
    assertThat(expressions).hasSize(2);
    assertThat(expressions.get(0).is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(listLiteralTree.leftBracket()).isNotNull();
    assertThat(listLiteralTree.rightBracket()).isNotNull();
    assertThat(listLiteralTree.children()).hasSize(3);
  }


  @Test
  public void list_comprehension() {
    setRootRule(PythonGrammar.TEST);
    PyComprehensionExpressionTree comprehension =
      (PyComprehensionExpressionTree) parse("[x+y for x,y in [(42, 43)]]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    assertThat(comprehension.firstToken().value()).isEqualTo("[");
    assertThat(comprehension.lastToken().value()).isEqualTo("]");
    assertThat(comprehension.resultExpression().getKind()).isEqualTo(Tree.Kind.PLUS);
    assertThat(comprehension.children()).hasSize(2);
    PyComprehensionForTree forClause = comprehension.comprehensionFor();
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
    PyComprehensionExpressionTree comprehension =
      (PyComprehensionExpressionTree) parse("[x+1 for x in [42, 43] if x%2==0]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    PyComprehensionForTree forClause = comprehension.comprehensionFor();
    assertThat(forClause.nestedClause().getKind()).isEqualTo(Tree.Kind.COMP_IF);
    PyComprehensionIfTree ifClause = (PyComprehensionIfTree) forClause.nestedClause();
    assertThat(ifClause.ifToken().value()).isEqualTo("if");
    assertThat(ifClause.condition().getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(ifClause.nestedClause()).isNull();
    assertThat(ifClause.children()).hasSize(2);
  }

  @Test
  public void list_comprehension_with_nested_for() {
    setRootRule(PythonGrammar.TEST);
    PyComprehensionExpressionTree comprehension =
      (PyComprehensionExpressionTree) parse("[x+y for x in [42, 43] for y in ('a', 0)]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    PyComprehensionForTree forClause = comprehension.comprehensionFor();
    assertThat(forClause.iterable().getKind()).isEqualTo(Tree.Kind.LIST_LITERAL);
    assertThat(forClause.nestedClause().getKind()).isEqualTo(Tree.Kind.COMP_FOR);
  }

  @Test
  public void parenthesized_expression() {
    setRootRule(PythonGrammar.TEST);
    PyParenthesizedExpressionTree parenthesized = (PyParenthesizedExpressionTree) parse("(42)", treeMaker::expression);
    assertThat(parenthesized.getKind()).isEqualTo(Tree.Kind.PARENTHESIZED);
    assertThat(parenthesized.children()).hasSize(3);
    assertThat(parenthesized.leftParenthesis().value()).isEqualTo("(");
    assertThat(parenthesized.rightParenthesis().value()).isEqualTo(")");
    assertThat(parenthesized.expression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);

    parenthesized = (PyParenthesizedExpressionTree) parse("(yield 42)", treeMaker::expression);
    assertThat(parenthesized.expression().getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);
  }


  @Test
  public void generator_expression() {
    setRootRule(PythonGrammar.TEST);
    PyComprehensionExpressionTree generator = (PyComprehensionExpressionTree) parse("(x*x for x in range(10))", treeMaker::expression);
    assertThat(generator.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);
    assertThat(generator.children()).hasSize(2);
    assertThat(generator.firstToken().value()).isEqualTo("(");
    assertThat(generator.lastToken().value()).isEqualTo(")");
    assertThat(generator.resultExpression().getKind()).isEqualTo(Tree.Kind.MULTIPLICATION);
    assertThat(generator.comprehensionFor().iterable().getKind()).isEqualTo(Tree.Kind.CALL_EXPR);

    setRootRule(PythonGrammar.CALL_EXPR);
    PyCallExpressionTree call = (PyCallExpressionTree) parse("foo(x*x for x in range(10))", treeMaker::expression);
    assertThat(call.arguments()).hasSize(1);
    PyExpressionTree firstArg = call.arguments().get(0).expression();
    assertThat(firstArg.getKind()).isEqualTo(Tree.Kind.GENERATOR_EXPR);

    call = (PyCallExpressionTree) parse("foo((x*x for x in range(10)))", treeMaker::expression);
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
    PyTupleTree empty = parseTuple("()");
    assertThat(empty.getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(empty.elements()).isEmpty();
    assertThat(empty.commas()).isEmpty();
    assertThat(empty.leftParenthesis().value()).isEqualTo("(");
    assertThat(empty.rightParenthesis().value()).isEqualTo(")");
    assertThat(empty.children()).hasSize(2);

    PyTupleTree singleValue = parseTuple("(a,)");
    assertThat(singleValue.elements()).extracting(Tree::getKind).containsExactly(Tree.Kind.NAME);
    assertThat(singleValue.commas()).extracting(PyToken::value).containsExactly(",");
    assertThat(singleValue.children()).hasSize(4);

    assertThat(parseTuple("(a,b)").elements()).hasSize(2);
  }

  private PyTupleTree parseTuple(String code) {
    setRootRule(PythonGrammar.TEST);
    PyTupleTree tuple = (PyTupleTree) parse(code, treeMaker::expression);
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
    PyExpressionTree exp = parse("not 1", treeMaker::expression);
    assertThat(exp).isInstanceOf(PyUnaryExpressionTree.class);
    assertThat(exp.getKind()).isEqualTo(Tree.Kind.NOT);
    assertThat(((PyUnaryExpressionTree) exp).expression().is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
  }

  @Test
  public void conditional_expression() {
    setRootRule(PythonGrammar.TEST);
    PyConditionalExpressionTree conditionalExpressionTree = (PyConditionalExpressionTree) parse("1 if condition else 2", treeMaker::expression);
    assertThat(conditionalExpressionTree.firstToken().value()).isEqualTo("1");
    assertThat(conditionalExpressionTree.lastToken().value()).isEqualTo("2");
    assertThat(conditionalExpressionTree.ifKeyword().value()).isEqualTo("if");
    assertThat(conditionalExpressionTree.elseKeyword().value()).isEqualTo("else");
    assertThat(conditionalExpressionTree.condition().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(conditionalExpressionTree.trueExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(conditionalExpressionTree.falseExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);

    PyConditionalExpressionTree nestedConditionalExpressionTree =
      (PyConditionalExpressionTree) parse("1 if x else 2 if y else 3", treeMaker::expression);
    assertThat(nestedConditionalExpressionTree.condition().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(nestedConditionalExpressionTree.trueExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    PyExpressionTree nestedConditionalExpr = nestedConditionalExpressionTree.falseExpression();
    assertThat(nestedConditionalExpr.firstToken().value()).isEqualTo("2");
    assertThat(nestedConditionalExpr.lastToken().value()).isEqualTo("3");
    assertThat(nestedConditionalExpr.getKind()).isEqualTo(Tree.Kind.CONDITIONAL_EXPR);
  }

  @Test
  public void dictionary_literal() {
    setRootRule(PythonGrammar.ATOM);
    PyDictionaryLiteralTree tree = (PyDictionaryLiteralTree) parse("{'key': 'value'}", treeMaker::expression);
    assertThat(tree.firstToken().value()).isEqualTo("{");
    assertThat(tree.lastToken().value()).isEqualTo("}");
    assertThat(tree.getKind()).isEqualTo(Tree.Kind.DICTIONARY_LITERAL);
    assertThat(tree.elements()).hasSize(1);
    PyKeyValuePairTree keyValuePair = tree.elements().iterator().next();
    assertThat(keyValuePair.getKind()).isEqualTo(Tree.Kind.KEY_VALUE_PAIR);
    assertThat(keyValuePair.key().getKind()).isEqualTo(Tree.Kind.STRING_LITERAL);
    assertThat(keyValuePair.colon().value()).isEqualTo(":");
    assertThat(keyValuePair.value().getKind()).isEqualTo(Tree.Kind.STRING_LITERAL);
    assertThat(tree.children()).hasSize(1);

    tree = (PyDictionaryLiteralTree) parse("{'key': 'value', 'key2': 'value2'}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);

    tree = (PyDictionaryLiteralTree) parse("{** var}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(1);
    keyValuePair = tree.elements().iterator().next();
    assertThat(keyValuePair.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(keyValuePair.starStarToken().value()).isEqualTo("**");

    tree = (PyDictionaryLiteralTree) parse("{** var, key: value}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);
  }

  @Test
  public void dict_comprehension() {
    setRootRule(PythonGrammar.TEST);
    PyDictCompExpressionTree comprehension =
      (PyDictCompExpressionTree) parse("{x-1:y+1 for x,y in [(42,43)]}", treeMaker::expression);
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
    PySetLiteralTree tree = (PySetLiteralTree) parse("{ x }", treeMaker::expression);
    assertThat(tree.firstToken().value()).isEqualTo("{");
    assertThat(tree.lastToken().value()).isEqualTo("}");
    assertThat(tree.getKind()).isEqualTo(Tree.Kind.SET_LITERAL);
    assertThat(tree.elements()).hasSize(1);
    PyExpressionTree element = tree.elements().iterator().next();
    assertThat(element.getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(tree.lCurlyBrace().value()).isEqualTo("{");
    assertThat(tree.rCurlyBrace().value()).isEqualTo("}");
    assertThat(tree.commas()).hasSize(0);
    assertThat(tree.children()).hasSize(1);

    tree = (PySetLiteralTree) parse("{ x, y }", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);

    tree = (PySetLiteralTree) parse("{ *x }", treeMaker::expression);
    assertThat(tree.elements()).hasSize(1);
    element = tree.elements().iterator().next();
    assertThat(element.getKind()).isEqualTo(Tree.Kind.STARRED_EXPR);
  }

  @Test
  public void set_comprehension() {
    setRootRule(PythonGrammar.TEST);
    PyComprehensionExpressionTree comprehension =
      (PyComprehensionExpressionTree) parse("{x-1 for x in [42, 43]}", treeMaker::expression);
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
    PyReprExpressionTree reprExpressionTree = (PyReprExpressionTree) parse("`1`", treeMaker::expression);
    assertThat(reprExpressionTree.getKind()).isEqualTo(Tree.Kind.REPR);
    assertThat(reprExpressionTree.firstToken().value()).isEqualTo("`");
    assertThat(reprExpressionTree.lastToken().value()).isEqualTo("`");
    assertThat(reprExpressionTree.openingBacktick().value()).isEqualTo("`");
    assertThat(reprExpressionTree.closingBacktick().value()).isEqualTo("`");
    assertThat(reprExpressionTree.expressionList().expressions()).hasSize(1);
    assertThat(reprExpressionTree.expressionList().expressions().get(0).getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(reprExpressionTree.children()).hasSize(3);

    reprExpressionTree = (PyReprExpressionTree) parse("`x, y`", treeMaker::expression);
    assertThat(reprExpressionTree.getKind()).isEqualTo(Tree.Kind.REPR);
    assertThat(reprExpressionTree.expressionList().expressions()).hasSize(2);
    assertThat(reprExpressionTree.expressionList().expressions().get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(reprExpressionTree.expressionList().expressions().get(1).getKind()).isEqualTo(Tree.Kind.NAME);
  }

  @Test
  public void ellipsis_expression() {
    setRootRule(PythonGrammar.ATOM);
    PyEllipsisExpressionTree ellipsisExpressionTree = (PyEllipsisExpressionTree) parse("...", treeMaker::expression);
    assertThat(ellipsisExpressionTree.getKind()).isEqualTo(Tree.Kind.ELLIPSIS);
    assertThat(ellipsisExpressionTree.ellipsis()).extracting(PyToken::value).containsExactly(".", ".", ".");
    assertThat(ellipsisExpressionTree.children()).hasSize(3);
  }

  @Test
  public void none_expression() {
    setRootRule(PythonGrammar.ATOM);
    PyNoneExpressionTree noneExpressionTree = (PyNoneExpressionTree) parse("None", treeMaker::expression);
    assertThat(noneExpressionTree.getKind()).isEqualTo(Tree.Kind.NONE);
    assertThat(noneExpressionTree.none().value()).isEqualTo("None");
    assertThat(noneExpressionTree.children()).hasSize(1);
  }

  @Test
  public void variables() {
    setRootRule(PythonGrammar.ATOM);
    PyNameTree name = (PyNameTree) parse("foo", treeMaker::expression);
    assertThat(name.isVariable()).isTrue();

    setRootRule(PythonGrammar.ATTRIBUTE_REF);
    PyQualifiedExpressionTree qualifiedExpressionTree = (PyQualifiedExpressionTree) parse("a.b", treeMaker::expression);
    assertThat(qualifiedExpressionTree.name().isVariable()).isFalse();

    setRootRule(PythonGrammar.FUNCDEF);
    PyFunctionDefTree functionDefTree = parse("def func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.name().isVariable()).isFalse();
  }

  private void assertUnaryExpression(String operator, Tree.Kind kind) {
    setRootRule(PythonGrammar.EXPR);
    PyExpressionTree parse = parse(operator+"1", treeMaker::expression);
    assertThat(parse.is(kind)).isTrue();
    PyUnaryExpressionTree unary = (PyUnaryExpressionTree) parse;
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
