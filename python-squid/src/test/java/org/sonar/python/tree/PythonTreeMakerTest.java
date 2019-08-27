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
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyAssertStatementTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyBreakStatementTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyContinueStatementTree;
import org.sonar.python.api.tree.PyDelStatementTree;
import org.sonar.python.api.tree.PyElseStatementTree;
import org.sonar.python.api.tree.PyExceptClauseTree;
import org.sonar.python.api.tree.PyExecStatementTree;
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
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyNonlocalStatementTree;
import org.sonar.python.api.tree.PyPassStatementTree;
import org.sonar.python.api.tree.PyPrintStatementTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.PyRaiseStatementTree;
import org.sonar.python.api.tree.PyReturnStatementTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyTryStatementTree;
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

    testData.forEach((c,clazz) -> {
      PyFileInputTree pyTree = parse(c, treeMaker::fileInput);
      PyStatementListTree statementList = pyTree.statements();
      assertThat(statementList.statements()).hasSize(1);
      assertThat(statementList.statements().get(0)).as(c).isInstanceOf(clazz);
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


    pyIfStatementTree = parse("if x: pass\nelse: pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    PyElseStatementTree elseBranch = pyIfStatementTree.elseBranch();
    assertThat(elseBranch).isNotNull();
    assertThat(elseBranch.elseKeyword().getValue()).isEqualTo("else");
    assertThat(elseBranch.body().statements()).hasSize(1);


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

    pyIfStatementTree = parse("if x:\n pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    assertThat(pyIfStatementTree.body().statements()).hasSize(1);

    pyIfStatementTree = parse("if x:\n pass\n pass", treeMaker::ifStatement);
    assertThat(pyIfStatementTree.body().statements()).hasSize(2);
  }

  @Test
  public void printStatement() {
    setRootRule(PythonGrammar.PRINT_STMT);
    AstNode astNode = p.parse("print 'foo'");
    PyPrintStatementTree printStmt = treeMaker.printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().getValue()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(1);

    astNode = p.parse("print 'foo', 'bar'");
    printStmt = treeMaker.printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().getValue()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(2);

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

    astNode = p.parse("exec 'foo' in globals");
    execStatement = treeMaker.execStatement(astNode);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().getValue()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNotNull();
    assertThat(execStatement.localsExpression()).isNull();

    astNode = p.parse("exec 'foo' in globals, locals");
    execStatement = treeMaker.execStatement(astNode);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().getValue()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNotNull();
    assertThat(execStatement.localsExpression()).isNotNull();

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

    astNode = p.parse("assert x, y");
    assertStatement = treeMaker.assertStatement(astNode);
    assertThat(assertStatement).isNotNull();
    assertThat(assertStatement.assertKeyword().getValue()).isEqualTo("assert");
    assertThat(assertStatement.expressions()).hasSize(2);
  }

  @Test
  public void passStatement() {
    setRootRule(PythonGrammar.PASS_STMT);
    AstNode astNode = p.parse("pass");
    PyPassStatementTree passStatement = treeMaker.passStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.passKeyword().getValue()).isEqualTo("pass");
  }

  @Test
  public void delStatement() {
    setRootRule(PythonGrammar.DEL_STMT);
    AstNode astNode = p.parse("del foo");
    PyDelStatementTree passStatement = treeMaker.delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().getValue()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);

    astNode = p.parse("del foo, bar");
    passStatement = treeMaker.delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().getValue()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(2);

    astNode = p.parse("del *foo");
    passStatement = treeMaker.delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().getValue()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);
  }

  @Test
  public void returnStatement() {
    setRootRule(PythonGrammar.RETURN_STMT);
    AstNode astNode = p.parse("return foo");
    PyReturnStatementTree returnStatement = treeMaker.returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().getValue()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(1);

    astNode = p.parse("return foo, bar");
    returnStatement = treeMaker.returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().getValue()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(2);

    astNode = p.parse("return");
    returnStatement = treeMaker.returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().getValue()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(0);
  }

  @Test
  public void yieldStatement() {
    setRootRule(PythonGrammar.YIELD_STMT);
    AstNode astNode = p.parse("yield foo");
    PyYieldStatementTree yieldStatement = treeMaker.yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    PyYieldExpressionTree yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.expressions()).hasSize(1);

    astNode = p.parse("yield foo, bar");
    yieldStatement = treeMaker.yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.yieldKeyword().getValue()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword()).isNull();
    assertThat(yieldExpression.expressions()).hasSize(2);

    astNode = p.parse("yield from foo");
    yieldStatement = treeMaker.yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.yieldKeyword().getValue()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword().getValue()).isEqualTo("from");
    assertThat(yieldExpression.expressions()).hasSize(1);

    astNode = p.parse("yield");
    yieldStatement = treeMaker.yieldStatement(astNode);
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

    astNode = p.parse("raise foo, bar");
    raiseStatement = treeMaker.raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).hasSize(2);

    astNode = p.parse("raise foo from bar");
    raiseStatement = treeMaker.raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword().getValue()).isEqualTo("from");
    assertThat(raiseStatement.fromExpression()).isNotNull();
    assertThat(raiseStatement.expressions()).hasSize(1);

    astNode = p.parse("raise");
    raiseStatement = treeMaker.raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).isEmpty();
  }

  @Test
  public void breakStatement() {
    setRootRule(PythonGrammar.BREAK_STMT);
    AstNode astNode = p.parse("break");
    PyBreakStatementTree breakStatement = treeMaker.breakStatement(astNode);
    assertThat(breakStatement).isNotNull();
    assertThat(breakStatement.breakKeyword().getValue()).isEqualTo("break");
  }

  @Test
  public void continueStatement() {
    setRootRule(PythonGrammar.CONTINUE_STMT);
    AstNode astNode = p.parse("continue");
    PyContinueStatementTree continueStatement = treeMaker.continueStatement(astNode);
    assertThat(continueStatement).isNotNull();
    assertThat(continueStatement.continueKeyword().getValue()).isEqualTo("continue");
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

    astNode = p.parse("import foo.bar");
    importStatement = (PyImportNameTree) treeMaker.importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().getValue()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(2);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importedName1.dottedName().names().get(1).name()).isEqualTo("bar");

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

    astNode = p.parse("from .foo import f");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).getValue()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");

    astNode = p.parse("from ..foo import f");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(2);
    assertThat(importStatement.dottedPrefixForModule().get(0).getValue()).isEqualTo(".");
    assertThat(importStatement.dottedPrefixForModule().get(1).getValue()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");

    astNode = p.parse("from . import f");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).getValue()).isEqualTo(".");
    assertThat(importStatement.module()).isNull();

    astNode = p.parse("from foo import f as g");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.importedNames()).hasSize(1);
    aliasedNameTree = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree.asKeyword().getValue()).isEqualTo("as");
    assertThat(aliasedNameTree.alias().name()).isEqualTo("g");
    assertThat(aliasedNameTree.dottedName().names().get(0).name()).isEqualTo("f");

    astNode = p.parse("from foo import f as g, h");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.importedNames()).hasSize(2);
    PyAliasedNameTree aliasedNameTree1 = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree1.asKeyword().getValue()).isEqualTo("as");
    assertThat(aliasedNameTree1.alias().name()).isEqualTo("g");
    assertThat(aliasedNameTree1.dottedName().names().get(0).name()).isEqualTo("f");

    PyAliasedNameTree aliasedNameTree2 = importStatement.importedNames().get(1);
    assertThat(aliasedNameTree2.asKeyword()).isNull();
    assertThat(aliasedNameTree2.alias()).isNull();
    assertThat(aliasedNameTree2.dottedName().names().get(0).name()).isEqualTo("h");

    astNode = p.parse("from foo import *");
    importStatement = (PyImportFromTree) treeMaker.importStatement(astNode);
    assertThat(importStatement.importedNames()).isNull();
    assertThat(importStatement.isWildcardImport()).isTrue();
    assertThat(importStatement.wildcard().getValue()).isEqualTo("*");
  }

  @Test
  public void globalStatement() {
    setRootRule(PythonGrammar.GLOBAL_STMT);
    AstNode astNode = p.parse("global foo");
    PyGlobalStatementTree globalStatement = treeMaker.globalStatement(astNode);
    assertThat(globalStatement.globalKeyword().getValue()).isEqualTo("global");
    assertThat(globalStatement.variables()).hasSize(1);
    assertThat(globalStatement.variables().get(0).name()).isEqualTo("foo");

    astNode = p.parse("global foo, bar");
    globalStatement = treeMaker.globalStatement(astNode);
    assertThat(globalStatement.globalKeyword().getValue()).isEqualTo("global");
    assertThat(globalStatement.variables()).hasSize(2);
    assertThat(globalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(globalStatement.variables().get(1).name()).isEqualTo("bar");
  }

  @Test
  public void nonlocalStatement() {
    setRootRule(PythonGrammar.NONLOCAL_STMT);
    AstNode astNode = p.parse("nonlocal foo");
    PyNonlocalStatementTree nonlocalStatement = treeMaker.nonlocalStatement(astNode);
    assertThat(nonlocalStatement.nonlocalKeyword().getValue()).isEqualTo("nonlocal");
    assertThat(nonlocalStatement.variables()).hasSize(1);
    assertThat(nonlocalStatement.variables().get(0).name()).isEqualTo("foo");

    astNode = p.parse("nonlocal foo, bar");
    nonlocalStatement = treeMaker.nonlocalStatement(astNode);
    assertThat(nonlocalStatement.nonlocalKeyword().getValue()).isEqualTo("nonlocal");
    assertThat(nonlocalStatement.variables()).hasSize(2);
    assertThat(nonlocalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(nonlocalStatement.variables().get(1).name()).isEqualTo("bar");
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
    // TODO
    assertThat(functionDefTree.typedArgs()).isNull();
    assertThat(functionDefTree.decorators()).isNull();
    assertThat(functionDefTree.asyncKeyword()).isNull();
    assertThat(functionDefTree.colon()).isNull();
    assertThat(functionDefTree.defKeyword()).isNull();
    assertThat(functionDefTree.dash()).isNull();
    assertThat(functionDefTree.gt()).isNull();
    assertThat(functionDefTree.leftPar()).isNull();
    assertThat(functionDefTree.rightPar()).isNull();

  }

  @Test
  public void classdef_statement() {
    setRootRule(PythonGrammar.CLASSDEF);
    AstNode astNode = p.parse("class clazz: pass");
    PyClassDefTree classDefTree = treeMaker.classDefStatement(astNode);
    assertThat(classDefTree.name()).isNotNull();
    assertThat(classDefTree.name().name()).isEqualTo("clazz");
    assertThat(classDefTree.body().statements()).hasSize(1);
    assertThat(classDefTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(classDefTree.args()).isNull();
    assertThat(classDefTree.decorators()).isNull();
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

    astNode = p.parse("for foo in bar:\n  pass\nelse:\n  pass");
    pyForStatementTree = treeMaker.forStatement(astNode);
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body().statements()).hasSize(1);
    assertThat(pyForStatementTree.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody().statements()).hasSize(1);
    assertThat(pyForStatementTree.elseBody().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();

    // TODO
    assertThat(pyForStatementTree.forKeyword()).isNull();
    assertThat(pyForStatementTree.inKeyword()).isNull();
    assertThat(pyForStatementTree.colon()).isNull();
    assertThat(pyForStatementTree.elseKeyword()).isNull();
    assertThat(pyForStatementTree.elseColon()).isNull();
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

    astNode = p.parse("while foo:\n  pass\nelse:\n  pass");
    whileStatement = treeMaker.whileStatement(astNode);
    assertThat(whileStatement.condition()).isNotNull();
    assertThat(whileStatement.body().statements()).hasSize(1);
    assertThat(whileStatement.body().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(whileStatement.elseBody().statements()).hasSize(1);
    assertThat(whileStatement.elseBody().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();

    // TODO
    assertThat(whileStatement.whileKeyword()).isNull();
    assertThat(whileStatement.colon()).isNull();
    assertThat(whileStatement.elseKeyword()).isNull();
    assertThat(whileStatement.elseColon()).isNull();

  }

  @Test
  public void expression_statement() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("'foo'");
    PyExpressionStatementTree expressionStatement = treeMaker.expressionStatement(astNode);
    assertThat(expressionStatement.expressions()).hasSize(1);

    astNode = p.parse("'foo', 'bar'");
    expressionStatement = treeMaker.expressionStatement(astNode);
    assertThat(expressionStatement.expressions()).hasSize(2);
  }

  @Test
  public void assignement_statement() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("x = y");
    PyAssignmentStatementTree pyAssignmentStatement = treeMaker.assignment(astNode);
    PyNameTree assigned = (PyNameTree) pyAssignmentStatement.assignedValues().get(0);
    PyNameTree lhs = (PyNameTree) pyAssignmentStatement.lhsExpressions().get(0).expressions().get(0);
    assertThat(assigned.name()).isEqualTo("y");
    assertThat(lhs.name()).isEqualTo("x");

    astNode = p.parse("x = y = z");
    pyAssignmentStatement = treeMaker.assignment(astNode);
    assigned = (PyNameTree) pyAssignmentStatement.assignedValues().get(0);
    lhs = (PyNameTree) pyAssignmentStatement.lhsExpressions().get(0).expressions().get(0);
    PyNameTree lhs2 = (PyNameTree) pyAssignmentStatement.lhsExpressions().get(1).expressions().get(0);
    assertThat(assigned.name()).isEqualTo("z");
    assertThat(lhs.name()).isEqualTo("x");
    assertThat(lhs2.name()).isEqualTo("y");
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

    astNode = p.parse("try: pass\nexcept Error: pass\nexcept Error: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.elseClause()).isNull();
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.exceptClauses()).hasSize(2);

    astNode = p.parse("try: pass\nexcept Error: pass\nfinally: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.elseClause()).isNull();
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    assertThat(tryStatement.finallyClause()).isNotNull();
    assertThat(tryStatement.finallyClause().finallyKeyword().getValue()).isEqualTo("finally");
    assertThat(tryStatement.finallyClause().body().statements()).hasSize(1);

    astNode = p.parse("try: pass\nexcept Error: pass\nelse: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    assertThat(tryStatement.finallyClause()).isNull();
    assertThat(tryStatement.elseClause().elseKeyword().getValue()).isEqualTo("else");
    assertThat(tryStatement.elseClause().body().statements()).hasSize(1);

    astNode = p.parse("try: pass\nexcept Error as e: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    PyExceptClauseTree exceptClause = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause.asKeyword().getValue()).isEqualTo("as");
    assertThat(exceptClause.commaToken()).isNull();
    assertThat(exceptClause.exceptionInstance()).isNotNull();

    astNode = p.parse("try: pass\nexcept Error, e: pass");
    tryStatement = treeMaker.tryStatement(astNode);
    assertThat(tryStatement.tryKeyword().getValue()).isEqualTo("try");
    assertThat(tryStatement.exceptClauses()).hasSize(1);
    exceptClause = tryStatement.exceptClauses().get(0);
    assertThat(exceptClause.asKeyword()).isNull();
    assertThat(exceptClause.commaToken().getValue()).isEqualTo(",");
    assertThat(exceptClause.exceptionInstance()).isNotNull();
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

    PyWithStatementTree withStatement = parse("async with foo : pass", treeMaker::withStatement);
    assertThat(withStatement.isAsync()).isTrue();
    assertThat(withStatement.asyncKeyword().getValue()).isEqualTo("async");
    PyWithItemTree pyWithItemTree = withStatement.withItems().get(0);
    assertThat(pyWithItemTree.test()).isNotNull();
    assertThat(pyWithItemTree.as()).isNull();
    assertThat(pyWithItemTree.expression()).isNull();
    assertThat(withStatement.statements().statements()).hasSize(1);
    assertThat(withStatement.statements().statements().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
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
  }

  @Test
  public void verify_expected_expression() {
    Map<String, Class<? extends Tree>> testData = new HashMap<>();
    testData.put("foo", PyNameTree.class);
    testData.put("foo.bar", PyQualifiedExpressionTree.class);
    testData.put("foo()", PyCallExpressionTree.class);

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

    callExpression = parse("foo(x, y)", treeMaker::callExpression);
    assertThat(callExpression.arguments()).hasSize(2);
    PyNameTree firstArg = (PyNameTree) callExpression.arguments().get(0).expression();
    PyNameTree sndArg = (PyNameTree) callExpression.arguments().get(1).expression();
    assertThat(firstArg.name()).isEqualTo("x");
    assertThat(sndArg.name()).isEqualTo("y");
    name = (PyNameTree) callExpression.callee();
    assertThat(name.name()).isEqualTo("foo");

    callExpression = parse("foo.bar()", treeMaker::callExpression);
    assertThat(callExpression.arguments()).isEmpty();
    PyQualifiedExpressionTree callee = (PyQualifiedExpressionTree) callExpression.callee();
    assertThat(callee.name().name()).isEqualTo("bar");
    assertThat(((PyNameTree) callee.qualifier()).name()).isEqualTo("foo");
  }

  @Test
  public void attributeRef_expression() {
    setRootRule(PythonGrammar.ATTRIBUTE_REF);
    PyQualifiedExpressionTree qualifiedExpression = parse("foo.bar", treeMaker::qualifiedExpression);
    assertThat(qualifiedExpression.name().name()).isEqualTo("bar");
    PyExpressionTree qualifier = qualifiedExpression.qualifier();
    assertThat(qualifier).isInstanceOf(PyNameTree.class);
    assertThat(((PyNameTree) qualifier).name()).isEqualTo("foo");

    qualifiedExpression = parse("foo.bar.baz", treeMaker::qualifiedExpression);
    assertThat(qualifiedExpression.name().name()).isEqualTo("baz");
    assertThat(qualifiedExpression.qualifier()).isInstanceOf(PyQualifiedExpressionTree.class);
    PyQualifiedExpressionTree qualExpr = (PyQualifiedExpressionTree) qualifiedExpression.qualifier();
    assertThat(qualExpr.name().name()).isEqualTo("bar");
    assertThat(qualExpr.qualifier()).isInstanceOf(PyNameTree.class);
    PyNameTree name = (PyNameTree) qualExpr.qualifier();
    assertThat(name.name()).isEqualTo("foo");
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

    argumentTree = parse("*foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNull();
    assertThat(argumentTree.keywordArgument()).isNull();
    name = (PyNameTree) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNotNull();
    assertThat(argumentTree.starStarToken()).isNull();

    argumentTree = parse("**foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNull();
    assertThat(argumentTree.keywordArgument()).isNull();
    name = (PyNameTree) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNull();
    assertThat(argumentTree.starStarToken()).isNotNull();

    argumentTree = parse("bar=foo", treeMaker::argument);
    assertThat(argumentTree.equalToken()).isNotNull();
    PyNameTree keywordArgument = argumentTree.keywordArgument();
    assertThat(keywordArgument.name()).isEqualTo("bar");
    name = (PyNameTree) argumentTree.expression();
    assertThat(name.name()).isEqualTo("foo");
    assertThat(argumentTree.starToken()).isNull();
    assertThat(argumentTree.starStarToken()).isNull();
  }

  @Test
  public void binary_expressions() {
    setRootRule(PythonGrammar.EXPR);

    PyBinaryExpressionTree simplePlus = binaryExpression("a + b");
    assertThat(simplePlus.leftOperand()).isInstanceOf(PyNameTree.class);
    assertThat(simplePlus.operator().getValue()).isEqualTo("+");
    assertThat(simplePlus.rightOperand()).isInstanceOf(PyNameTree.class);
    assertThat(simplePlus.getKind()).isEqualTo(Tree.Kind.PLUS);

    PyBinaryExpressionTree compoundPlus = binaryExpression("a + b - c");
    assertThat(compoundPlus.leftOperand()).isInstanceOf(PyBinaryExpressionTree.class);
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

    setRootRule(PythonGrammar.TEST);
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

  private <T> T parse(String code, Function<AstNode, T> func) {
    return func.apply(p.parse(code));
  }
}
