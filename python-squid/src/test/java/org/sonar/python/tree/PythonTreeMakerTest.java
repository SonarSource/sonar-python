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
import com.sonar.sslr.api.Token;
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
import org.sonar.python.api.tree.PyComprehensionForTree;
import org.sonar.python.api.tree.PyComprehensionIfTree;
import org.sonar.python.api.tree.PyConditionalExpressionTree;
import org.sonar.python.api.tree.PyContinueStatementTree;
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
import org.sonar.python.api.tree.PyListOrSetCompExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyNoneExpressionTree;
import org.sonar.python.api.tree.PyNonlocalStatementTree;
import org.sonar.python.api.tree.PyNumericLiteralTree;
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
import org.sonar.python.api.tree.PyTryStatementTree;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.PyTypedArgumentTree;
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
    assertThat(pyTree.docstring().getValue()).isEqualTo("\"\"\"\n" +
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
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.body().statements()).hasSize(1);
    assertThat(pyIfStatementTree.children()).hasSize(3);


    pyIfStatementTree = parse("if x: pass\nelse: pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    PyElseStatementTree elseBranch = pyIfStatementTree.elseBranch();
    assertThat(elseBranch).isNotNull();
    assertThat(elseBranch.elseKeyword().getValue()).isEqualTo("else");
    assertThat(elseBranch.body().statements()).hasSize(1);
    assertThat(pyIfStatementTree.children()).hasSize(3);


    pyIfStatementTree = parse("if x: pass\nelif y: pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.elifBranches()).hasSize(1);
    PyIfStatementTree elif = pyIfStatementTree.elifBranches().get(0);
    assertThat(elif.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(elif.isElif()).isTrue();
    assertThat(elif.elseBranch()).isNull();
    assertThat(elif.elifBranches()).isEmpty();
    assertThat(elif.body().statements()).hasSize(1);
    assertThat(pyIfStatementTree.children()).hasSize(4);

    pyIfStatementTree = parse("if x:\n pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
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
    assertThat(pyFileInputTree.body().tokens()).isEqualTo(parseTree.getFirstChild(PythonGrammar.SUITE).getTokens());
  }

  @Test
  public void printStatement() {
    setRootRule(PythonGrammar.PRINT_STMT);
    AstNode astNode = p.parse("print 'foo'");
    PyPrintStatementTree printStmt = treeMaker.printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().getValue()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(1);
    assertThat(printStmt.children()).hasSize(1);

    astNode = p.parse("print 'foo', 'bar'");
    printStmt = treeMaker.printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().getValue()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(2);
    assertThat(printStmt.children()).hasSize(2);

    astNode = p.parse("print >> 'foo'");
    printStmt = treeMaker.printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().getValue()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(1);
  }

  @Test
  public void execStatement() {
    setRootRule(PythonGrammar.EXEC_STMT);
    AstNode astNode = p.parse("exec 'foo'");
    PyExecStatementTree execStatement = treeMaker.execStatement(astNode);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().getValue()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNull();
    assertThat(execStatement.localsExpression()).isNull();
    assertThat(execStatement.children()).hasSize(3);

    astNode = p.parse("exec 'foo' in globals");
    execStatement = treeMaker.execStatement(astNode);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().getValue()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNotNull();
    assertThat(execStatement.localsExpression()).isNull();
    assertThat(execStatement.children()).hasSize(3);

    astNode = p.parse("exec 'foo' in globals, locals");
    execStatement = treeMaker.execStatement(astNode);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().getValue()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNotNull();
    assertThat(execStatement.localsExpression()).isNotNull();
    assertThat(execStatement.children()).hasSize(3);

    // TODO: exec stmt should parse exec ('foo', globals, locals); see https://docs.python.org/2/reference/simple_stmts.html#exec
  }

  @Test
  public void assertStatement() {
    setRootRule(PythonGrammar.ASSERT_STMT);
    AstNode astNode = p.parse("assert x");
    PyAssertStatementTree assertStatement = treeMaker.assertStatement(astNode);
    assertThat(assertStatement).isNotNull();
    assertThat(assertStatement.assertKeyword().getValue()).isEqualTo("assert");
    assertThat(assertStatement.expressions()).hasSize(1);
    assertThat(assertStatement.children()).hasSize(1);

    astNode = p.parse("assert x, y");
    assertStatement = treeMaker.assertStatement(astNode);
    assertThat(assertStatement).isNotNull();
    assertThat(assertStatement.assertKeyword().getValue()).isEqualTo("assert");
    assertThat(assertStatement.expressions()).hasSize(2);
    assertThat(assertStatement.children()).hasSize(2);
  }

  @Test
  public void passStatement() {
    setRootRule(PythonGrammar.PASS_STMT);
    AstNode astNode = p.parse("pass");
    PyPassStatementTree passStatement = treeMaker.passStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.passKeyword().getValue()).isEqualTo("pass");
    assertThat(passStatement.children()).isEmpty();
  }

  @Test
  public void delStatement() {
    setRootRule(PythonGrammar.DEL_STMT);
    AstNode astNode = p.parse("del foo");
    PyDelStatementTree passStatement = treeMaker.delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().getValue()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);
    assertThat(passStatement.children()).hasSize(1);


    astNode = p.parse("del foo, bar");
    passStatement = treeMaker.delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().getValue()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(2);
    assertThat(passStatement.children()).hasSize(2);

    astNode = p.parse("del *foo");
    passStatement = treeMaker.delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().getValue()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);
    assertThat(passStatement.children()).hasSize(1);
  }

  @Test
  public void returnStatement() {
    setRootRule(PythonGrammar.RETURN_STMT);
    AstNode astNode = p.parse("return foo");
    PyReturnStatementTree returnStatement = treeMaker.returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().getValue()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(1);
    assertThat(returnStatement.children()).hasSize(1);

    astNode = p.parse("return foo, bar");
    returnStatement = treeMaker.returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().getValue()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(2);
    assertThat(returnStatement.children()).hasSize(2);

    astNode = p.parse("return");
    returnStatement = treeMaker.returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().getValue()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(0);
    assertThat(returnStatement.children()).isEmpty();
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
    assertThat(yieldExpression.children()).hasSize(1);

    astNode = p.parse("yield foo, bar");
    yieldStatement = treeMaker.yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    assertThat(yieldStatement.children()).hasSize(1);
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.yieldKeyword().getValue()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword()).isNull();
    assertThat(yieldExpression.expressions()).hasSize(2);
    assertThat(yieldExpression.children()).hasSize(2);

    astNode = p.parse("yield from foo");
    yieldStatement = treeMaker.yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    assertThat(yieldStatement.children()).hasSize(1);
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.yieldKeyword().getValue()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword().getValue()).isEqualTo("from");
    assertThat(yieldExpression.expressions()).hasSize(1);
    assertThat(yieldExpression.children()).hasSize(1);

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
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).hasSize(1);
    assertThat(raiseStatement.children()).hasSize(2);

    astNode = p.parse("raise foo, bar");
    raiseStatement = treeMaker.raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).hasSize(2);
    assertThat(raiseStatement.children()).hasSize(3);

    astNode = p.parse("raise foo from bar");
    raiseStatement = treeMaker.raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword().getValue()).isEqualTo("from");
    assertThat(raiseStatement.fromExpression()).isNotNull();
    assertThat(raiseStatement.expressions()).hasSize(1);
    assertThat(raiseStatement.children()).hasSize(2);

    astNode = p.parse("raise");
    raiseStatement = treeMaker.raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
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
    assertThat(breakStatement.breakKeyword().getValue()).isEqualTo("break");
    assertThat(breakStatement.children()).isEmpty();
  }

  @Test
  public void continueStatement() {
    setRootRule(PythonGrammar.CONTINUE_STMT);
    AstNode astNode = p.parse("continue");
    PyContinueStatementTree continueStatement = treeMaker.continueStatement(astNode);
    assertThat(continueStatement).isNotNull();
    assertThat(continueStatement.continueKeyword().getValue()).isEqualTo("continue");
    assertThat(continueStatement.children()).isEmpty();
  }

  @Test
  public void importStatement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    AstNode astNode = p.parse("import foo");
    PyImportNameTree importStatement = (PyImportNameTree) treeMaker.importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().getValue()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    PyAliasedNameTree importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.children()).hasSize(1);

    astNode = p.parse("import foo as f");
    importStatement = (PyImportNameTree) treeMaker.importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().getValue()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importedName1.asKeyword().getValue()).isEqualTo("as");
    assertThat(importedName1.alias().name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(1);

    astNode = p.parse("import foo.bar");
    importStatement = (PyImportNameTree) treeMaker.importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().getValue()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(2);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importedName1.dottedName().names().get(1).name()).isEqualTo("bar");
    assertThat(importStatement.children()).hasSize(1);

    astNode = p.parse("import foo, bar");
    importStatement = (PyImportNameTree) treeMaker.importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().getValue()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(2);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    PyAliasedNameTree importedName2 = importStatement.modules().get(1);
    assertThat(importedName2.dottedName().names()).hasSize(1);
    assertThat(importedName2.dottedName().names().get(0).name()).isEqualTo("bar");
    assertThat(importStatement.children()).hasSize(2);
  }

  @Test
  public void importFromStatement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    AstNode astNode = p.parse("from foo import f");
    PyImportFromTree importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().getValue()).isEqualTo("import");
    assertThat(importStatement.dottedPrefixForModule()).isEmpty();
    assertThat(importStatement.fromKeyword().getValue()).isEqualTo("from");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.isWildcardImport()).isFalse();
    assertThat(importStatement.wildcard()).isNull();
    assertThat(importStatement.importedNames()).hasSize(1);
    PyAliasedNameTree aliasedNameTree = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree.asKeyword()).isNull();
    assertThat(aliasedNameTree.alias()).isNull();
    assertThat(aliasedNameTree.dottedName().names().get(0).name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("from .foo import f");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).getValue()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("from ..foo import f");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(2);
    assertThat(importStatement.dottedPrefixForModule().get(0).getValue()).isEqualTo(".");
    assertThat(importStatement.dottedPrefixForModule().get(1).getValue()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("from . import f");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).getValue()).isEqualTo(".");
    assertThat(importStatement.module()).isNull();
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("from foo import f as g");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.importedNames()).hasSize(1);
    aliasedNameTree = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree.asKeyword().getValue()).isEqualTo("as");
    assertThat(aliasedNameTree.alias().name()).isEqualTo("g");
    assertThat(aliasedNameTree.dottedName().names().get(0).name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(2);

    astNode = p.parse("from foo import f as g, h");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.importedNames()).hasSize(2);
    PyAliasedNameTree aliasedNameTree1 = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree1.asKeyword().getValue()).isEqualTo("as");
    assertThat(aliasedNameTree1.alias().name()).isEqualTo("g");
    assertThat(aliasedNameTree1.dottedName().names().get(0).name()).isEqualTo("f");
    assertThat(importStatement.children()).hasSize(3);

    PyAliasedNameTree aliasedNameTree2 = importStatement.importedNames().get(1);
    assertThat(aliasedNameTree2.asKeyword()).isNull();
    assertThat(aliasedNameTree2.alias()).isNull();
    assertThat(aliasedNameTree2.dottedName().names().get(0).name()).isEqualTo("h");

    astNode = p.parse("from foo import *");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.importedNames()).isEmpty();
    assertThat(importStatement.isWildcardImport()).isTrue();
    assertThat(importStatement.wildcard().getValue()).isEqualTo("*");
    assertThat(importStatement.children()).hasSize(1);
  }

  @Test
  public void globalStatement() {
    setRootRule(PythonGrammar.GLOBAL_STMT);
    AstNode astNode = p.parse("global foo");
    PyGlobalStatementTree globalStatement = treeMaker.globalStatement(astNode);
    assertThat(globalStatement.globalKeyword().getValue()).isEqualTo("global");
    assertThat(globalStatement.variables()).hasSize(1);
    assertThat(globalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(globalStatement.children()).hasSize(1);

    astNode = p.parse("global foo, bar");
    globalStatement = treeMaker.globalStatement(astNode);
    assertThat(globalStatement.globalKeyword().getValue()).isEqualTo("global");
    assertThat(globalStatement.variables()).hasSize(2);
    assertThat(globalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(globalStatement.variables().get(1).name()).isEqualTo("bar");
    assertThat(globalStatement.children()).hasSize(2);
  }

  @Test
  public void nonlocalStatement() {
    setRootRule(PythonGrammar.NONLOCAL_STMT);
    AstNode astNode = p.parse("nonlocal foo");
    PyNonlocalStatementTree nonlocalStatement = treeMaker.nonlocalStatement(astNode);
    assertThat(nonlocalStatement.nonlocalKeyword().getValue()).isEqualTo("nonlocal");
    assertThat(nonlocalStatement.variables()).hasSize(1);
    assertThat(nonlocalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(nonlocalStatement.children()).hasSize(1);

    astNode = p.parse("nonlocal foo, bar");
    nonlocalStatement = treeMaker.nonlocalStatement(astNode);
    assertThat(nonlocalStatement.nonlocalKeyword().getValue()).isEqualTo("nonlocal");
    assertThat(nonlocalStatement.variables()).hasSize(2);
    assertThat(nonlocalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(nonlocalStatement.variables().get(1).name()).isEqualTo("bar");
    assertThat(nonlocalStatement.children()).hasSize(2);
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
    assertThat(functionDefTree.children()).hasSize(3);
    // TODO
    assertThat(functionDefTree.typedArgs()).isNull();
    assertThat(functionDefTree.isMethodDefinition()).isFalse();
    assertThat(functionDefTree.docstring()).isNull();
    // TODO
    assertThat(functionDefTree.decorators()).isNull();
    assertThat(functionDefTree.asyncKeyword()).isNull();
    assertThat(functionDefTree.colon()).isNull();
    assertThat(functionDefTree.defKeyword()).isNull();
    assertThat(functionDefTree.dash()).isNull();
    assertThat(functionDefTree.gt()).isNull();
    assertThat(functionDefTree.leftPar()).isNull();
    assertThat(functionDefTree.rightPar()).isNull();

    functionDefTree = parse("def func(x): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.typedArgs().arguments()).hasSize(1);

    functionDefTree = parse("def func(x, y): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.typedArgs().arguments()).hasSize(2);

    functionDefTree = parse("def func(x = 'foo', y): pass", treeMaker::funcDefStatement);
    List<PyTypedArgumentTree> args = functionDefTree.typedArgs().arguments();
    assertThat(args).hasSize(2);
    assertThat(args.get(0).expression().is(Tree.Kind.STRING_LITERAL)).isTrue();

    functionDefTree = parse("def func(x : int, y): pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.typedArgs().arguments()).hasSize(2);

    functionDefTree = parse("def func(x : int, y):\n  \"\"\"\n" +
      "This is a function docstring\n" +
      "\"\"\"\n  pass", treeMaker::funcDefStatement);
    assertThat(functionDefTree.docstring().getValue()).isEqualTo("\"\"\"\n" +
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
    assertThat(classDefTree.name().name()).isEqualTo("clazz");
    assertThat(classDefTree.body().statements()).hasSize(1);
    assertThat(classDefTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(classDefTree.args().is(Tree.Kind.ARG_LIST)).isTrue();
    assertThat(classDefTree.args().children()).hasSize(1);
    assertThat(classDefTree.args().arguments().get(0).is(Tree.Kind.ARGUMENT)).isTrue();
    assertThat(classDefTree.decorators()).isNull();

    astNode = p.parse("class clazz:\n  def foo(): pass");
    classDefTree = treeMaker.classDefStatement(astNode);
    PyFunctionDefTree funcDef = (PyFunctionDefTree) classDefTree.body().statements().get(0);
    assertThat(funcDef.isMethodDefinition()).isTrue();

    astNode = p.parse("class ClassWithDocstring:\n" +
      "\t\"\"\"This is a docstring\"\"\"\n" +
      "\tpass");
    classDefTree = treeMaker.classDefStatement(astNode);
    assertThat(classDefTree.docstring().getValue()).isEqualTo("\"\"\"This is a docstring\"\"\"");
    assertThat(classDefTree.children()).hasSize(3);
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
    assertThat(pyForStatementTree.children()).hasSize(4);

    astNode = p.parse("for foo in bar:\n  pass\nelse:\n  pass");
    pyForStatementTree = treeMaker.forStatement(astNode);
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body().statements()).hasSize(1);
    assertThat(pyForStatementTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody().statements()).hasSize(1);
    assertThat(pyForStatementTree.elseBody().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.children()).hasSize(4);

    assertThat(pyForStatementTree.forKeyword().getValue()).isEqualTo("for");
    assertThat(pyForStatementTree.inKeyword().getValue()).isEqualTo("in");
    assertThat(pyForStatementTree.colon().getValue()).isEqualTo(":");
    assertThat(pyForStatementTree.elseKeyword().getValue()).isEqualTo("else");
    assertThat(pyForStatementTree.elseColon().getValue()).isEqualTo(":");
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
    assertThat(whileStatement.children()).hasSize(3);

    astNode = p.parse("while foo:\n  pass\nelse:\n  pass");
    whileStatement = treeMaker.whileStatement(astNode);
    assertThat(whileStatement.condition()).isNotNull();
    assertThat(whileStatement.body().statements()).hasSize(1);
    assertThat(whileStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.elseBody().statements()).hasSize(1);
    assertThat(whileStatement.elseBody().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.children()).hasSize(3);

    assertThat(whileStatement.whileKeyword().getValue()).isEqualTo("while");
    assertThat(whileStatement.colon().getValue()).isEqualTo(":");
    assertThat(whileStatement.elseKeyword().getValue()).isEqualTo("else");
    assertThat(whileStatement.elseColon().getValue()).isEqualTo(":");

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
    PyNameTree assigned = (PyNameTree) pyAssignmentStatement.assignedValue();
    PyNameTree lhs = (PyNameTree) pyAssignmentStatement.lhsExpressions().get(0).expressions().get(0);
    assertThat(assigned.name()).isEqualTo("y");
    assertThat(lhs.name()).isEqualTo("x");
    assertThat(pyAssignmentStatement.children()).hasSize(2);

    astNode = p.parse("x = y = z");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.equalTokens()).hasSize(2);
    assertThat(pyAssignmentStatement.children()).hasSize(3);
    assigned = (PyNameTree) pyAssignmentStatement.assignedValue();
    lhs = (PyNameTree) pyAssignmentStatement.lhsExpressions().get(0).expressions().get(0);
    PyNameTree lhs2 = (PyNameTree) pyAssignmentStatement.lhsExpressions().get(1).expressions().get(0);
    assertThat(assigned.name()).isEqualTo("z");
    assertThat(lhs.name()).isEqualTo("x");
    assertThat(lhs2.name()).isEqualTo("y");

    astNode = p.parse("a,b = x");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.children()).hasSize(2);
    assigned = (PyNameTree) pyAssignmentStatement.assignedValue();
    List<PyExpressionTree> expressions = pyAssignmentStatement.lhsExpressions().get(0).expressions();
    assertThat(assigned.name()).isEqualTo("x");
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(expressions.get(1).getKind()).isEqualTo(Tree.Kind.NAME);

    astNode = p.parse("x = a,b");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.children()).hasSize(2);
    expressions = pyAssignmentStatement.lhsExpressions().get(0).expressions();
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyAssignmentStatement.assignedValue().getKind()).isEqualTo(Tree.Kind.TUPLE);

    astNode = p.parse("x = yield 1");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.children()).hasSize(2);
    expressions = pyAssignmentStatement.lhsExpressions().get(0).expressions();
    assertThat(expressions.get(0).getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyAssignmentStatement.assignedValue().getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);

    // FIXME: lhs expression list shouldn't allow yield expressions. We need to change the grammar
    astNode = p.parse("x = yield 1 = y");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assertThat(pyAssignmentStatement.children()).hasSize(3);
    List<PyExpressionListTree> lhsExpressions = pyAssignmentStatement.lhsExpressions();
    assertThat(lhsExpressions.get(1).expressions().get(0).getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);
    assertThat(pyAssignmentStatement.assignedValue().getKind()).isEqualTo(Tree.Kind.NAME);
  }

  @Test
  public void annotated_assignment() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x : string = 1");
    PyAnnotatedAssignmentTree annAssign = treeMaker.annotatedAssignment(astNode);
    assertThat(annAssign.getKind()).isEqualTo(Tree.Kind.ANNOTATED_ASSIGNMENT);
    assertThat(annAssign.children()).hasSize(3);
    assertThat(annAssign.variable().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((PyNameTree) annAssign.variable()).name()).isEqualTo("x");
    assertThat(annAssign.assignedValue().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(annAssign.equalToken().getValue()).isEqualTo("=");
    assertThat(annAssign.annotation().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(((PyNameTree) annAssign.annotation()).name()).isEqualTo("string");
    assertThat(annAssign.colonToken().getValue()).isEqualTo(":");

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
    assertThat(pyCompoundAssignmentStatement.children()).hasSize(2);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().getValue()).isEqualTo("+=");
    assertThat(pyCompoundAssignmentStatement.lhsExpression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyCompoundAssignmentStatement.rhsExpression().getKind()).isEqualTo(Tree.Kind.NAME);

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    astNode = p.parse("x,y,z += 1");
    pyCompoundAssignmentStatement = treeMaker.compoundAssignment(astNode);
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.children()).hasSize(2);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().getValue()).isEqualTo("+=");
    assertThat(pyCompoundAssignmentStatement.lhsExpression().getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(pyCompoundAssignmentStatement.rhsExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    astNode = p.parse("x += yield y");
    pyCompoundAssignmentStatement = treeMaker.compoundAssignment(astNode);
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.children()).hasSize(2);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().getValue()).isEqualTo("+=");
    assertThat(pyCompoundAssignmentStatement.lhsExpression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(pyCompoundAssignmentStatement.rhsExpression().getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);

    astNode = p.parse("x *= z");
    pyCompoundAssignmentStatement = treeMaker.compoundAssignment(astNode);
    assertThat(pyCompoundAssignmentStatement.getKind()).isEqualTo(Tree.Kind.COMPOUND_ASSIGNMENT);
    assertThat(pyCompoundAssignmentStatement.compoundAssignmentToken().getValue()).isEqualTo("*=");
  }

  @Test
  public void try_statement() {
    setRootRule(PythonGrammar.TRY_STMT);
    AstNode astNode = p.parse("try: pass\nexcept Error: pass");
    PyTryStatementTree tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.body().statements()).hasSize(1);
    assertThat(tryStatement.elseClause()).isNull();
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    assertThat(tryStatement.exceptClauses().get(0).exceptKeyword().getValue()).isEqualTo("except");
    assertThat(tryStatement.exceptClauses().get(0).body().statements()).hasSize(1);
    assertThat(tryStatement.children()).hasSize(4);


    astNode = p.parse("try: pass\nexcept Error: pass\nexcept Error: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.elseClause()).isNull();
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.exceptClauses()).hasSize(2);
    assertThat(tryStatement.children()).hasSize(5);

    astNode = p.parse("try: pass\nexcept Error: pass\nfinally: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.elseClause()).isNull();
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    assertThat(tryStatement.finallyClause()).isNotNull();
    assertThat(tryStatement.finallyClause().finallyKeyword().getValue()).isEqualTo("finally");
    assertThat(tryStatement.finallyClause().body().statements()).hasSize(1);
    assertThat(tryStatement.children()).hasSize(4);

    astNode = p.parse("try: pass\nexcept Error: pass\nelse: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.elseClause().elseKeyword().getValue()).isEqualTo("else");
    assertThat(tryStatement.elseClause().body().statements()).hasSize(1);
    assertThat(tryStatement.children()).hasSize(4);

    astNode = p.parse("try: pass\nexcept Error as e: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    PyExceptClauseTree exceptClause = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause.asKeyword().getValue()).isEqualTo("as");
    assertThat(exceptClause.commaToken()).isNull();
    assertThat(exceptClause.exceptionInstance()).isNotNull();
    assertThat(tryStatement.children()).hasSize(4);

    astNode = p.parse("try: pass\nexcept Error, e: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    exceptClause = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause.asKeyword()).isNull();
    assertThat(exceptClause.commaToken().getValue()).isEqualTo(",");
    assertThat(exceptClause.exceptionInstance()).isNotNull();
    assertThat(tryStatement.children()).hasSize(4);
  }

  @Test
  public void async_statement() {
    setRootRule(PythonGrammar.ASYNC_STMT);
    AstNode astNode = p.parse("async for foo in bar: pass");
    PyForStatementTree pyForStatementTree = new PythonTreeMaker().forStatement(astNode);
    assertThat(pyForStatementTree.isAsync()).isTrue();
    assertThat(pyForStatementTree.asyncKeyword().getValue()).isEqualTo("async");
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body().statements()).hasSize(1);
    assertThat(pyForStatementTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody()).isNull();
    assertThat(pyForStatementTree.children()).hasSize(4);

    PyWithStatementTree withStatement = parse("async with foo : pass", treeMaker::withStatement);
    assertThat(withStatement.isAsync()).isTrue();
    assertThat(withStatement.asyncKeyword().getValue()).isEqualTo("async");
    PyWithItemTree pyWithItemTree = withStatement.withItems().get(0);
    assertThat(pyWithItemTree.test()).isNotNull();
    assertThat(pyWithItemTree.as()).isNull();
    assertThat(pyWithItemTree.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(2);
  }

  @Test
  public void with_statement() {
    setRootRule(PythonGrammar.WITH_STMT);
    PyWithStatementTree withStatement = parse("with foo : pass", treeMaker::withStatement);
    assertThat(withStatement.isAsync()).isFalse();
    assertThat(withStatement.asyncKeyword()).isNull();
    assertThat(withStatement.withItems()).hasSize(1);
    PyWithItemTree pyWithItemTree = withStatement.withItems().get(0);
    assertThat(pyWithItemTree.test()).isNotNull();
    assertThat(pyWithItemTree.as()).isNull();
    assertThat(pyWithItemTree.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(2);


    withStatement = parse("with foo as bar, qix : pass", treeMaker::withStatement);
    assertThat(withStatement.withItems()).hasSize(2);
    pyWithItemTree = withStatement.withItems().get(0);
    assertThat(pyWithItemTree.test()).isNotNull();
    assertThat(pyWithItemTree.as()).isNotNull();
    assertThat(pyWithItemTree.expression()).isNotNull();
    pyWithItemTree = withStatement.withItems().get(1);
    assertThat(pyWithItemTree.test()).isNotNull();
    assertThat(pyWithItemTree.as()).isNull();
    assertThat(pyWithItemTree.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(withStatement.children()).hasSize(3);
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
    assertThat(callExpression.arguments()).isEmpty();
    PyNameTree name = (PyNameTree) callExpression.callee();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(1);

    callExpression = parse("foo(x, y)", treeMaker::callExpression);
    assertThat(callExpression.arguments()).hasSize(2);
    PyNameTree firstArg = (PyNameTree) callExpression.arguments().get(0).expression();
    PyNameTree sndArg = (PyNameTree) callExpression.arguments().get(1).expression();
    assertThat(firstArg.name()).isEqualTo("x");
    assertThat(sndArg.name()).isEqualTo("y");
    name = (PyNameTree) callExpression.callee();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(3);

    callExpression = parse("foo.bar()", treeMaker::callExpression);
    assertThat(callExpression.arguments()).isEmpty();
    PyQualifiedExpressionTree callee = (PyQualifiedExpressionTree) callExpression.callee();
    assertThat(callee.name().name()).isEqualTo("bar");
    assertThat(((PyNameTree) callee.qualifier()).name()).isEqualTo("foo");
    assertThat(callExpression.children()).hasSize(1);
  }

  @Test
  public void combinations_with_call_expressions() {
    setRootRule(PythonGrammar.TEST);

    PyCallExpressionTree nestingCall = (PyCallExpressionTree) parse("foo('a').bar(42)", treeMaker::expression);
    assertThat(nestingCall.arguments()).extracting(t -> t.expression().getKind()).containsExactly(Tree.Kind.NUMERIC_LITERAL);
    PyQualifiedExpressionTree callee = (PyQualifiedExpressionTree) nestingCall.callee();
    assertThat(callee.name().name()).isEqualTo("bar");
    assertThat(callee.qualifier().getKind()).isEqualTo(Tree.Kind.CALL_EXPR);

    PyCallExpressionTree callOnSubscription = (PyCallExpressionTree) parse("a[42](arg)", treeMaker::expression);
    PySubscriptionExpressionTree subscription = (PySubscriptionExpressionTree) callOnSubscription.callee();
    assertThat(((PyNameTree) subscription.object()).name()).isEqualTo("a");
    assertThat(subscription.subscripts().expressions()).extracting(Tree::getKind).containsExactly(Tree.Kind.NUMERIC_LITERAL);
    assertThat(((PyNameTree) callOnSubscription.arguments().get(0).expression()).name()).isEqualTo("arg");
  }

  @Test
  public void attributeRef_expression() {
    setRootRule(PythonGrammar.ATTRIBUTE_REF);
    PyQualifiedExpressionTree qualifiedExpression = parse("foo.bar", treeMaker::qualifiedExpression);
    assertThat(qualifiedExpression.name().name()).isEqualTo("bar");
    PyExpressionTree qualifier = qualifiedExpression.qualifier();
    assertThat(qualifier).isInstanceOf(PyNameTree.class);
    assertThat(((PyNameTree) qualifier).name()).isEqualTo("foo");
    assertThat(qualifiedExpression.children()).hasSize(2);

    qualifiedExpression = parse("foo.bar.baz", treeMaker::qualifiedExpression);
    assertThat(qualifiedExpression.name().name()).isEqualTo("baz");
    assertThat(qualifiedExpression.qualifier()).isInstanceOf(PyQualifiedExpressionTree.class);
    PyQualifiedExpressionTree qualExpr = (PyQualifiedExpressionTree) qualifiedExpression.qualifier();
    assertThat(qualExpr.name().name()).isEqualTo("bar");
    assertThat(qualExpr.qualifier()).isInstanceOf(PyNameTree.class);
    PyNameTree name = (PyNameTree) qualExpr.qualifier();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(qualifiedExpression.children()).hasSize(2);
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
    assertThat(argumentTree.children()).hasSize(2);

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
    assertThat(argumentTree.children()).hasSize(2);
  }

  @Test
  public void binary_expressions() {
    setRootRule(PythonGrammar.TEST);

    PyBinaryExpressionTree simplePlus = binaryExpression("a + b");
    assertThat(simplePlus.leftOperand()).isInstanceOf(PyNameTree.class);
    assertThat(simplePlus.operator().getValue()).isEqualTo("+");
    assertThat(simplePlus.rightOperand()).isInstanceOf(PyNameTree.class);
    assertThat(simplePlus.getKind()).isEqualTo(Tree.Kind.PLUS);
    assertThat(simplePlus.children()).hasSize(2);

    PyBinaryExpressionTree compoundPlus = binaryExpression("a + b - c");
    assertThat(compoundPlus.leftOperand()).isInstanceOf(PyBinaryExpressionTree.class);
    assertThat(compoundPlus.children()).hasSize(2);
    assertThat(compoundPlus.operator().getValue()).isEqualTo("-");
    assertThat(compoundPlus.rightOperand()).isInstanceOf(PyNameTree.class);
    assertThat(compoundPlus.getKind()).isEqualTo(Tree.Kind.MINUS);
    PyBinaryExpressionTree compoundPlusLeft = (PyBinaryExpressionTree) compoundPlus.leftOperand();
    assertThat(compoundPlusLeft.operator().getValue()).isEqualTo("+");

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
    assertThat(in.operator().getValue()).isEqualTo("in");
    assertThat(in.leftOperand().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(in.rightOperand().getKind()).isEqualTo(Tree.Kind.LIST_LITERAL);
    assertThat(in.notToken()).isNull();

    PyInExpressionTree notIn = (PyInExpressionTree) binaryExpression("1 not in [a]");
    assertThat(notIn.getKind()).isEqualTo(Tree.Kind.IN);
    assertThat(notIn.operator().getValue()).isEqualTo("in");
    assertThat(notIn.notToken()).isNotNull();
  }

  @Test
  public void is_expressions() {
    setRootRule(PythonGrammar.TEST);

    PyIsExpressionTree in = (PyIsExpressionTree) binaryExpression("a is 1");
    assertThat(in.getKind()).isEqualTo(Tree.Kind.IS);
    assertThat(in.operator().getValue()).isEqualTo("is");
    assertThat(in.leftOperand().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(in.rightOperand().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(in.notToken()).isNull();

    PyIsExpressionTree notIn = (PyIsExpressionTree) binaryExpression("a is not 1");
    assertThat(notIn.getKind()).isEqualTo(Tree.Kind.IS);
    assertThat(notIn.operator().getValue()).isEqualTo("is");
    assertThat(notIn.notToken()).isNotNull();
  }

  @Test
  public void starred_expression() {
    setRootRule(PythonGrammar.STAR_EXPR);
    PyStarredExpressionTree starred = (PyStarredExpressionTree) parse("*a", treeMaker::expression);
    assertThat(starred.getKind()).isEqualTo(Tree.Kind.STARRED_EXPR);
    assertThat(starred.starToken().getValue()).isEqualTo("*");
    assertThat(starred.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(starred.children()).hasSize(1);
  }

  @Test
  public void await_expression() {
    setRootRule(PythonGrammar.TEST);
    PyAwaitExpressionTree expr = (PyAwaitExpressionTree) parse("await x", treeMaker::expression);
    assertThat(expr.getKind()).isEqualTo(Tree.Kind.AWAIT);
    assertThat(expr.awaitToken().getValue()).isEqualTo("await");
    assertThat(expr.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(expr.children()).hasSize(1);

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
    assertThat(expr.leftBracket().getValue()).isEqualTo("[");
    assertThat(expr.rightBracket().getValue()).isEqualTo("]");
    assertThat(expr.children()).hasSize(2);

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
    assertThat(expr.leftBracket().getValue()).isEqualTo("[");
    assertThat(expr.rightBracket().getValue()).isEqualTo("]");
    assertThat(expr.children()).hasSize(2);
    assertThat(expr.sliceList().getKind()).isEqualTo(Tree.Kind.SLICE_LIST);
    assertThat(expr.sliceList().children()).hasSize(1);
    assertThat(expr.sliceList().slices().get(0).getKind()).isEqualTo(Tree.Kind.SLICE_ITEM);

    PySliceExpressionTree multipleSlices = (PySliceExpressionTree) parse("x[a, b:c, :]", treeMaker::expression);
    List<Tree> slices = multipleSlices.sliceList().slices();
    assertThat(slices).extracting(Tree::getKind).containsExactly(Tree.Kind.NAME, Tree.Kind.SLICE_ITEM, Tree.Kind.SLICE_ITEM);
    assertThat(multipleSlices.sliceList().separators()).extracting(Token::getValue).containsExactly(",", ",");
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
    assertThat(slice.boundSeparator().getValue()).isEqualTo(":");
    assertThat(slice.strideSeparator().getValue()).isEqualTo(":");
    assertThat(slice.children()).hasSize(3);

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
  }

  @Test
  public void lambda_expr() {
    setRootRule(PythonGrammar.LAMBDEF);
    PyLambdaExpressionTree lambdaExpressionTree = parse("lambda x: x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.expression()).isInstanceOf(PyNameTree.class);
    assertThat(lambdaExpressionTree.lambdaKeyword().getValue()).isEqualTo("lambda");
    assertThat(lambdaExpressionTree.colonToken().getValue()).isEqualTo(":");

    assertThat(lambdaExpressionTree.arguments().arguments()).hasSize(1);
    assertThat(lambdaExpressionTree.children()).hasSize(2);

    lambdaExpressionTree = parse("lambda x, y: x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.arguments().arguments()).hasSize(2);
    assertThat(lambdaExpressionTree.children()).hasSize(2);

    lambdaExpressionTree = parse("lambda x = 'foo': x", treeMaker::lambdaExpression);
    List<PyTypedArgumentTree> arguments = lambdaExpressionTree.arguments().arguments();
    assertThat(arguments).hasSize(1);
    assertThat(arguments.get(0).keywordArgument().name()).isEqualTo("x");
    assertThat(lambdaExpressionTree.children()).hasSize(2);

    lambdaExpressionTree = parse("lambda (x, y): x", treeMaker::lambdaExpression);
    assertThat(lambdaExpressionTree.arguments().arguments()).hasSize(1);
    assertThat(lambdaExpressionTree.children()).hasSize(2);
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
    assertStringLiteral("u\'plop\'", "plop");
    assertStringLiteral("b\"abcdef\"", "abcdef");
    assertStringLiteral("f\"\"\"Eric Idle\"\"\"", "Eric Idle");
    assertStringLiteral("fr'x={4*10}'", "x={4*10}");
    assertStringLiteral("f'He said his name is {name} and he is {age} years old.'", "He said his name is {name} and he is {age} years old.");
    assertStringLiteral("f'''He said his name is {name.upper()}\n    ...    and he is {6 * seven} years old.'''",
      "He said his name is {name.upper()}\n    ...    and he is {6 * seven} years old.");
  }

  private void assertStringLiteral(String fullValue, String trimmedQuoteValue) {
    PyExpressionTree parse = parse(fullValue, treeMaker::expression);
    assertThat(parse.is(Tree.Kind.STRING_LITERAL)).isTrue();
    PyStringLiteralTree stringLiteral = (PyStringLiteralTree) parse;
    assertThat(stringLiteral.stringElements()).hasSize(1);
    PyStringElementTree firstElement = stringLiteral.stringElements().get(0);
    assertThat(firstElement.value()).isEqualTo(fullValue);
    assertThat(firstElement.trimmedQuotesValue()).isEqualTo(trimmedQuoteValue);
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
    PyListLiteralTree listLiteralTree = (PyListLiteralTree) parse;
    List<PyExpressionTree> expressions = listLiteralTree.elements().expressions();
    assertThat(expressions).hasSize(2);
    assertThat(expressions.get(0).is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(listLiteralTree.leftBracket()).isNotNull();
    assertThat(listLiteralTree.rightBracket()).isNotNull();
    assertThat(listLiteralTree.children()).hasSize(1);
  }


  @Test
  public void list_comprehension() {
    setRootRule(PythonGrammar.TEST);
    PyListOrSetCompExpressionTree comprehension =
      (PyListOrSetCompExpressionTree) parse("[x+y for x,y in [(42, 43)]]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    assertThat(comprehension.resultExpression().getKind()).isEqualTo(Tree.Kind.PLUS);
    assertThat(comprehension.children()).hasSize(2);
    PyComprehensionForTree forClause = comprehension.comprehensionFor();
    assertThat(forClause.getKind()).isEqualTo(Tree.Kind.COMP_FOR);
    assertThat(forClause.forToken().getValue()).isEqualTo("for");
    assertThat(forClause.loopExpression().getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(forClause.inToken().getValue()).isEqualTo("in");
    assertThat(forClause.iterable().getKind()).isEqualTo(Tree.Kind.LIST_LITERAL);
    assertThat(forClause.nestedClause()).isNull();
    assertThat(forClause.children()).hasSize(3);
  }

  @Test
  public void list_comprehension_with_if() {
    setRootRule(PythonGrammar.TEST);
    PyListOrSetCompExpressionTree comprehension =
      (PyListOrSetCompExpressionTree) parse("[x+1 for x in [42, 43] if x%2==0]", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.LIST_COMPREHENSION);
    PyComprehensionForTree forClause = comprehension.comprehensionFor();
    assertThat(forClause.nestedClause().getKind()).isEqualTo(Tree.Kind.COMP_IF);
    PyComprehensionIfTree ifClause = (PyComprehensionIfTree) forClause.nestedClause();
    assertThat(ifClause.ifToken().getValue()).isEqualTo("if");
    assertThat(ifClause.condition().getKind()).isEqualTo(Tree.Kind.COMPARISON);
    assertThat(ifClause.nestedClause()).isNull();
    assertThat(ifClause.children()).hasSize(2);
  }

  @Test
  public void list_comprehension_with_nested_for() {
    setRootRule(PythonGrammar.TEST);
    PyListOrSetCompExpressionTree comprehension =
      (PyListOrSetCompExpressionTree) parse("[x+y for x in [42, 43] for y in ('a', 0)]", treeMaker::expression);
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
    assertThat(parenthesized.children()).hasSize(1);
    assertThat(parenthesized.leftParenthesis().getValue()).isEqualTo("(");
    assertThat(parenthesized.rightParenthesis().getValue()).isEqualTo(")");
    assertThat(parenthesized.expression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);

    parenthesized = (PyParenthesizedExpressionTree) parse("(yield 42)", treeMaker::expression);
    assertThat(parenthesized.expression().getKind()).isEqualTo(Tree.Kind.YIELD_EXPR);
  }

  @Test
  public void tuples() {
    PyTupleTree empty = parseTuple("()");
    assertThat(empty.getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(empty.elements()).isEmpty();
    assertThat(empty.commas()).isEmpty();
    assertThat(empty.leftParenthesis().getValue()).isEqualTo("(");
    assertThat(empty.rightParenthesis().getValue()).isEqualTo(")");
    assertThat(empty.children()).hasSize(0);

    PyTupleTree singleValue = parseTuple("(a,)");
    assertThat(singleValue.elements()).extracting(Tree::getKind).containsExactly(Tree.Kind.NAME);
    assertThat(singleValue.commas()).extracting(Token::getValue).containsExactly(",");
    assertThat(singleValue.children()).hasSize(1);

    assertThat(parseTuple("(a,b)").elements()).hasSize(2);
  }

  private PyTupleTree parseTuple(String code) {
    setRootRule(PythonGrammar.TEST);
    return (PyTupleTree) parse(code, treeMaker::expression);
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
    assertThat(conditionalExpressionTree.ifKeyword().getValue()).isEqualTo("if");
    assertThat(conditionalExpressionTree.elseKeyword().getValue()).isEqualTo("else");
    assertThat(conditionalExpressionTree.condition().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(conditionalExpressionTree.trueExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(conditionalExpressionTree.falseExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);

    PyConditionalExpressionTree nestedConditionalExpressionTree =
      (PyConditionalExpressionTree) parse("1 if x else 2 if y else 3", treeMaker::expression);
    assertThat(nestedConditionalExpressionTree.condition().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(nestedConditionalExpressionTree.trueExpression().getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(nestedConditionalExpressionTree.falseExpression().getKind()).isEqualTo(Tree.Kind.CONDITIONAL_EXPR);
  }

  @Test
  public void dictionary_literal() {
    setRootRule(PythonGrammar.ATOM);
    PyDictionaryLiteralTree tree = (PyDictionaryLiteralTree) parse("{'key': 'value'}", treeMaker::expression);
    assertThat(tree.getKind()).isEqualTo(Tree.Kind.DICTIONARY_LITERAL);
    assertThat(tree.elements()).hasSize(1);
    PyKeyValuePairTree keyValuePair = tree.elements().iterator().next();
    assertThat(keyValuePair.getKind()).isEqualTo(Tree.Kind.KEY_VALUE_PAIR);
    assertThat(keyValuePair.key().getKind()).isEqualTo(Tree.Kind.STRING_LITERAL);
    assertThat(keyValuePair.colon().getValue()).isEqualTo(":");
    assertThat(keyValuePair.value().getKind()).isEqualTo(Tree.Kind.STRING_LITERAL);
    assertThat(tree.children()).hasSize(1);

    tree = (PyDictionaryLiteralTree) parse("{'key': 'value', 'key2': 'value2'}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);

    tree = (PyDictionaryLiteralTree) parse("{** var}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(1);
    keyValuePair = tree.elements().iterator().next();
    assertThat(keyValuePair.expression().getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(keyValuePair.starStarToken().getValue()).isEqualTo("**");

    tree = (PyDictionaryLiteralTree) parse("{** var, key: value}", treeMaker::expression);
    assertThat(tree.elements()).hasSize(2);
  }

  @Test
  public void dict_comprehension() {
    setRootRule(PythonGrammar.TEST);
    PyDictCompExpressionTree comprehension =
      (PyDictCompExpressionTree) parse("{x-1:y+1 for x,y in [(42,43)]}", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.DICT_COMPREHENSION);
    assertThat(comprehension.colonToken().getValue()).isEqualTo(":");
    assertThat(comprehension.keyExpression().getKind()).isEqualTo(Tree.Kind.MINUS);
    assertThat(comprehension.valueExpression().getKind()).isEqualTo(Tree.Kind.PLUS);
    assertThat(comprehension.comprehensionFor().loopExpression().getKind()).isEqualTo(Tree.Kind.TUPLE);
    assertThat(comprehension.children()).hasSize(3);
    assertThat(comprehension.firstToken().getValue()).isEqualTo("{");
    assertThat(comprehension.lastToken().getValue()).isEqualTo("}");
  }

  @Test
  public void set_literal() {
    setRootRule(PythonGrammar.ATOM);
    PySetLiteralTree tree = (PySetLiteralTree) parse("{ x }", treeMaker::expression);
    assertThat(tree.getKind()).isEqualTo(Tree.Kind.SET_LITERAL);
    assertThat(tree.elements()).hasSize(1);
    PyExpressionTree element = tree.elements().iterator().next();
    assertThat(element.getKind()).isEqualTo(Tree.Kind.NAME);
    assertThat(tree.lCurlyBrace().getValue()).isEqualTo("{");
    assertThat(tree.rCurlyBrace().getValue()).isEqualTo("}");
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
    PyListOrSetCompExpressionTree comprehension =
      (PyListOrSetCompExpressionTree) parse("{x-1 for x in [42, 43]}", treeMaker::expression);
    assertThat(comprehension.getKind()).isEqualTo(Tree.Kind.SET_COMPREHENSION);
    assertThat(comprehension.resultExpression().getKind()).isEqualTo(Tree.Kind.MINUS);
    assertThat(comprehension.children()).hasSize(2);
    assertThat(comprehension.firstToken().getValue()).isEqualTo("{");
    assertThat(comprehension.lastToken().getValue()).isEqualTo("}");
  }

  @Test
  public void repr_expression() {
    setRootRule(PythonGrammar.ATOM);
    PyReprExpressionTree reprExpressionTree = (PyReprExpressionTree) parse("`1`", treeMaker::expression);
    assertThat(reprExpressionTree.getKind()).isEqualTo(Tree.Kind.REPR);
    assertThat(reprExpressionTree.openingBacktick().getValue()).isEqualTo("`");
    assertThat(reprExpressionTree.closingBacktick().getValue()).isEqualTo("`");
    assertThat(reprExpressionTree.expressionList().expressions()).hasSize(1);
    assertThat(reprExpressionTree.expressionList().expressions().get(0).getKind()).isEqualTo(Tree.Kind.NUMERIC_LITERAL);
    assertThat(reprExpressionTree.children()).hasSize(1);

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
    assertThat(ellipsisExpressionTree.ellipsis()).extracting(Token::getValue).containsExactly(".", ".", ".");
    assertThat(ellipsisExpressionTree.children()).isEmpty();
  }

  @Test
  public void none_expression() {
    setRootRule(PythonGrammar.ATOM);
    PyNoneExpressionTree noneExpressionTree = (PyNoneExpressionTree) parse("None", treeMaker::expression);
    assertThat(noneExpressionTree.getKind()).isEqualTo(Tree.Kind.NONE);
    assertThat(noneExpressionTree.none().getValue()).isEqualTo("None");
    assertThat(noneExpressionTree.children()).isEmpty();
  }

  private void assertUnaryExpression(String operator, Tree.Kind kind) {
    setRootRule(PythonGrammar.EXPR);
    PyExpressionTree parse = parse(operator+"1", treeMaker::expression);
    assertThat(parse.is(kind)).isTrue();
    PyUnaryExpressionTree unary = (PyUnaryExpressionTree) parse;
    assertThat(unary.expression().is(Tree.Kind.NUMERIC_LITERAL)).isTrue();
    assertThat(unary.operator().getValue()).isEqualTo(operator);
    assertThat(unary.children()).hasSize(1);
  }

  private <T extends Tree> T parse(String code, Function<AstNode, T> func) {
    T tree = func.apply(p.parse(code));
    // ensure every visit method of base tree visitor is called without errors
    BaseTreeVisitor visitor = new BaseTreeVisitor();
    tree.accept(visitor);
    return tree;
  }
}
