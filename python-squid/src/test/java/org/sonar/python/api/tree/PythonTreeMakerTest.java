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
package org.sonar.python.api.tree;

import com.sonar.sslr.api.AstNode;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;
import org.sonar.python.tree.PyWhileStatementTreeImpl;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonTreeMakerTest extends RuleTest {

  @Test
  public void fileInputTreeOnEmptyFile() {
    AstNode astNode = p.parse("");
    PyFileInputTree pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).isEmpty();
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
    testData.put("global foo", PyGlobalStatementTree.class);
    testData.put("while cond: pass", PyWhileStatementTree.class);


    testData.forEach((c,clazz) -> {
      AstNode astNode = p.parse(c);
      PyFileInputTree pyTree = new PythonTreeMaker().fileInput(astNode);
      assertThat(pyTree.statements()).hasSize(1);
      assertThat(pyTree.statements().get(0)).as(c).isInstanceOf(clazz);
    });
  }

  @Test
  public void IfStatement() {
    setRootRule(PythonGrammar.IF_STMT);
    AstNode astNode = p.parse("if x: pass");
    PyIfStatementTree pyIfStatementTree = new PythonTreeMaker().ifStatement(astNode);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.body()).hasSize(1);

    astNode = p.parse("if x: pass\nelse: pass");
    pyIfStatementTree = new PythonTreeMaker().ifStatement(astNode);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    PyElseStatementTree elseBranch = pyIfStatementTree.elseBranch();
    assertThat(elseBranch).isNotNull();
    assertThat(elseBranch.elseKeyword().getValue()).isEqualTo("else");
    assertThat(elseBranch.body()).hasSize(1);

    astNode = p.parse("if x: pass\nelif y: pass");
    pyIfStatementTree = new PythonTreeMaker().ifStatement(astNode);
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
    assertThat(elif.body()).hasSize(1);

    astNode = p.parse("if x:\n pass");
    pyIfStatementTree = new PythonTreeMaker().ifStatement(astNode);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.isElif()).isFalse();
    assertThat(pyIfStatementTree.elseBranch()).isNull();
    assertThat(pyIfStatementTree.elifBranches()).isEmpty();
    assertThat(pyIfStatementTree.body()).hasSize(1);
  }

  @Test
  public void printStatement() {
    setRootRule(PythonGrammar.PRINT_STMT);
    AstNode astNode = p.parse("print 'foo'");
    PyPrintStatementTree printStmt = new PythonTreeMaker().printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().getValue()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(1);

    astNode = p.parse("print 'foo', 'bar'");
    printStmt = new PythonTreeMaker().printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().getValue()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(2);

    astNode = p.parse("print >> 'foo'");
    printStmt = new PythonTreeMaker().printStatement(astNode);
    assertThat(printStmt).isNotNull();
    assertThat(printStmt.printKeyword().getValue()).isEqualTo("print");
    assertThat(printStmt.expressions()).hasSize(1);
  }

  @Test
  public void execStatement() {
    setRootRule(PythonGrammar.EXEC_STMT);
    AstNode astNode = p.parse("exec 'foo'");
    PyExecStatementTree execStatement = new PythonTreeMaker().execStatement(astNode);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().getValue()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNull();
    assertThat(execStatement.localsExpression()).isNull();

    astNode = p.parse("exec 'foo' in globals");
    execStatement = new PythonTreeMaker().execStatement(astNode);
    assertThat(execStatement).isNotNull();
    assertThat(execStatement.execKeyword().getValue()).isEqualTo("exec");
    assertThat(execStatement.expression()).isNotNull();
    assertThat(execStatement.globalsExpression()).isNotNull();
    assertThat(execStatement.localsExpression()).isNull();

    astNode = p.parse("exec 'foo' in globals, locals");
    execStatement = new PythonTreeMaker().execStatement(astNode);
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
    PyAssertStatementTree assertStatement = new PythonTreeMaker().assertStatement(astNode);
    assertThat(assertStatement).isNotNull();
    assertThat(assertStatement.assertKeyword().getValue()).isEqualTo("assert");
    assertThat(assertStatement.expressions()).hasSize(1);

    astNode = p.parse("assert x, y");
    assertStatement = new PythonTreeMaker().assertStatement(astNode);
    assertThat(assertStatement).isNotNull();
    assertThat(assertStatement.assertKeyword().getValue()).isEqualTo("assert");
    assertThat(assertStatement.expressions()).hasSize(2);
  }

  @Test
  public void passStatement() {
    setRootRule(PythonGrammar.PASS_STMT);
    AstNode astNode = p.parse("pass");
    PyPassStatementTree passStatement = new PythonTreeMaker().passStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.passKeyword().getValue()).isEqualTo("pass");
  }

  @Test
  public void delStatement() {
    setRootRule(PythonGrammar.DEL_STMT);
    AstNode astNode = p.parse("del foo");
    PyDelStatementTree passStatement = new PythonTreeMaker().delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().getValue()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);

    astNode = p.parse("del foo, bar");
    passStatement = new PythonTreeMaker().delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().getValue()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(2);

    astNode = p.parse("del *foo");
    passStatement = new PythonTreeMaker().delStatement(astNode);
    assertThat(passStatement).isNotNull();
    assertThat(passStatement.delKeyword().getValue()).isEqualTo("del");
    assertThat(passStatement.expressions()).hasSize(1);
  }

  @Test
  public void returnStatement() {
    setRootRule(PythonGrammar.RETURN_STMT);
    AstNode astNode = p.parse("return foo");
    PyReturnStatementTree returnStatement = new PythonTreeMaker().returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().getValue()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(1);

    astNode = p.parse("return foo, bar");
    returnStatement = new PythonTreeMaker().returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().getValue()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(2);

    astNode = p.parse("return");
    returnStatement = new PythonTreeMaker().returnStatement(astNode);
    assertThat(returnStatement).isNotNull();
    assertThat(returnStatement.returnKeyword().getValue()).isEqualTo("return");
    assertThat(returnStatement.expressions()).hasSize(0);
  }

  @Test
  public void yieldStatement() {
    setRootRule(PythonGrammar.YIELD_STMT);
    AstNode astNode = p.parse("yield foo");
    PyYieldStatementTree yieldStatement = new PythonTreeMaker().yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    PyYieldExpressionTree yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.expressions()).hasSize(1);

    astNode = p.parse("yield foo, bar");
    yieldStatement = new PythonTreeMaker().yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.yieldKeyword().getValue()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword()).isNull();
    assertThat(yieldExpression.expressions()).hasSize(2);

    astNode = p.parse("yield from foo");
    yieldStatement = new PythonTreeMaker().yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
    yieldExpression = yieldStatement.yieldExpression();
    assertThat(yieldExpression).isInstanceOf(PyYieldExpressionTree.class);
    assertThat(yieldExpression.yieldKeyword().getValue()).isEqualTo("yield");
    assertThat(yieldExpression.fromKeyword().getValue()).isEqualTo("from");
    assertThat(yieldExpression.expressions()).hasSize(1);

    astNode = p.parse("yield");
    yieldStatement = new PythonTreeMaker().yieldStatement(astNode);
    assertThat(yieldStatement).isNotNull();
  }

  @Test
  public void raiseStatement() {
    setRootRule(PythonGrammar.RAISE_STMT);
    AstNode astNode = p.parse("raise foo");
    PyRaiseStatementTree raiseStatement = new PythonTreeMaker().raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).hasSize(1);

    astNode = p.parse("raise foo, bar");
    raiseStatement = new PythonTreeMaker().raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword()).isNull();
    assertThat(raiseStatement.fromExpression()).isNull();
    assertThat(raiseStatement.expressions()).hasSize(2);

    astNode = p.parse("raise foo from bar");
    raiseStatement = new PythonTreeMaker().raiseStatement(astNode);
    assertThat(raiseStatement).isNotNull();
    assertThat(raiseStatement.raiseKeyword().getValue()).isEqualTo("raise");
    assertThat(raiseStatement.fromKeyword().getValue()).isEqualTo("from");
    assertThat(raiseStatement.fromExpression()).isNotNull();
    assertThat(raiseStatement.expressions()).hasSize(1);

    astNode = p.parse("raise");
    raiseStatement = new PythonTreeMaker().raiseStatement(astNode);
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
    PyBreakStatementTree breakStatement = new PythonTreeMaker().breakStatement(astNode);
    assertThat(breakStatement).isNotNull();
    assertThat(breakStatement.breakKeyword().getValue()).isEqualTo("break");
  }

  @Test
  public void continueStatement() {
    setRootRule(PythonGrammar.CONTINUE_STMT);
    AstNode astNode = p.parse("continue");
    PyContinueStatementTree continueStatement = new PythonTreeMaker().continueStatement(astNode);
    assertThat(continueStatement).isNotNull();
    assertThat(continueStatement.continueKeyword().getValue()).isEqualTo("continue");
  }

  @Test
  public void importStatement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    AstNode astNode = p.parse("import foo");
    PyImportNameTree importStatement = (PyImportNameTree) new PythonTreeMaker().importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().getValue()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    PyAliasedNameTree importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");

    astNode = p.parse("import foo as f");
    importStatement = (PyImportNameTree) new PythonTreeMaker().importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().getValue()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(1);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importedName1.asKeyword().getValue()).isEqualTo("as");
    assertThat(importedName1.alias().name()).isEqualTo("f");

    astNode = p.parse("import foo.bar");
    importStatement = (PyImportNameTree) new PythonTreeMaker().importStatement(astNode);
    assertThat(importStatement).isNotNull();
    assertThat(importStatement.importKeyword().getValue()).isEqualTo("import");
    assertThat(importStatement.modules()).hasSize(1);
    importedName1 = importStatement.modules().get(0);
    assertThat(importedName1.dottedName().names()).hasSize(2);
    assertThat(importedName1.dottedName().names().get(0).name()).isEqualTo("foo");
    assertThat(importedName1.dottedName().names().get(1).name()).isEqualTo("bar");

    astNode = p.parse("import foo, bar");
    importStatement = (PyImportNameTree) new PythonTreeMaker().importStatement(astNode);
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
    PyImportFromTree importStatement = (PyImportFromTree) new PythonTreeMaker().importStatement(astNode);
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
    importStatement = (PyImportFromTree) new PythonTreeMaker().importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).getValue()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");

    astNode = p.parse("from ..foo import f");
    importStatement = (PyImportFromTree) new PythonTreeMaker().importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(2);
    assertThat(importStatement.dottedPrefixForModule().get(0).getValue()).isEqualTo(".");
    assertThat(importStatement.dottedPrefixForModule().get(1).getValue()).isEqualTo(".");
    assertThat(importStatement.module().names().get(0).name()).isEqualTo("foo");

    astNode = p.parse("from . import f");
    importStatement = (PyImportFromTree) new PythonTreeMaker().importStatement(astNode);
    assertThat(importStatement.dottedPrefixForModule()).hasSize(1);
    assertThat(importStatement.dottedPrefixForModule().get(0).getValue()).isEqualTo(".");
    assertThat(importStatement.module()).isNull();

    astNode = p.parse("from foo import f as g");
    importStatement = (PyImportFromTree) new PythonTreeMaker().importStatement(astNode);
    assertThat(importStatement.importedNames()).hasSize(1);
    aliasedNameTree = importStatement.importedNames().get(0);
    assertThat(aliasedNameTree.asKeyword().getValue()).isEqualTo("as");
    assertThat(aliasedNameTree.alias().name()).isEqualTo("g");
    assertThat(aliasedNameTree.dottedName().names().get(0).name()).isEqualTo("f");

    astNode = p.parse("from foo import f as g, h");
    importStatement = (PyImportFromTree) new PythonTreeMaker().importStatement(astNode);
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
    importStatement = (PyImportFromTree) new PythonTreeMaker().importStatement(astNode);
    assertThat(importStatement.importedNames()).isNull();
    assertThat(importStatement.isWildcardImport()).isTrue();
    assertThat(importStatement.wildcard().getValue()).isEqualTo("*");
  }

  @Test
  public void globalStatement() {
    setRootRule(PythonGrammar.GLOBAL_STMT);
    AstNode astNode = p.parse("global foo");
    PyGlobalStatementTree globalStatement = new PythonTreeMaker().globalStatement(astNode);
    assertThat(globalStatement.globalKeyword().getValue()).isEqualTo("global");
    assertThat(globalStatement.variables()).hasSize(1);
    assertThat(globalStatement.variables().get(0).name()).isEqualTo("foo");

    astNode = p.parse("global foo, bar");
    globalStatement = new PythonTreeMaker().globalStatement(astNode);
    assertThat(globalStatement.globalKeyword().getValue()).isEqualTo("global");
    assertThat(globalStatement.variables()).hasSize(2);
    assertThat(globalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(globalStatement.variables().get(1).name()).isEqualTo("bar");
  }

  @Test
  public void nonlocalStatement() {
    setRootRule(PythonGrammar.NONLOCAL_STMT);
    AstNode astNode = p.parse("nonlocal foo");
    PyNonlocalStatementTree nonlocalStatement = new PythonTreeMaker().nonlocalStatement(astNode);
    assertThat(nonlocalStatement.nonlocalKeyword().getValue()).isEqualTo("nonlocal");
    assertThat(nonlocalStatement.variables()).hasSize(1);
    assertThat(nonlocalStatement.variables().get(0).name()).isEqualTo("foo");

    astNode = p.parse("nonlocal foo, bar");
    nonlocalStatement = new PythonTreeMaker().nonlocalStatement(astNode);
    assertThat(nonlocalStatement.nonlocalKeyword().getValue()).isEqualTo("nonlocal");
    assertThat(nonlocalStatement.variables()).hasSize(2);
    assertThat(nonlocalStatement.variables().get(0).name()).isEqualTo("foo");
    assertThat(nonlocalStatement.variables().get(1).name()).isEqualTo("bar");
  }

  @Test
  public void funcdef_statement() {
    setRootRule(PythonGrammar.FUNCDEF);
    AstNode astNode = p.parse("def func(): pass");
    PyFunctionDefTree functionDefTree = new PythonTreeMaker().funcDefStatement(astNode);
    assertThat(functionDefTree.name()).isNotNull();
    assertThat(functionDefTree.name().name()).isEqualTo("func");
    assertThat(functionDefTree.body()).hasSize(1);
    assertThat(functionDefTree.body().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(functionDefTree.typedArgs()).isNull();
    assertThat(functionDefTree.decorators()).isNull();

  }

  @Test
  public void classdef_statement() {
    setRootRule(PythonGrammar.CLASSDEF);
    AstNode astNode = p.parse("class clazz: pass");
    PyClassDefTree classDefTree = new PythonTreeMaker().classDefStatement(astNode);
    assertThat(classDefTree.name()).isNotNull();
    assertThat(classDefTree.name().name()).isEqualTo("clazz");
    assertThat(classDefTree.body()).hasSize(1);
    assertThat(classDefTree.body().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(classDefTree.args()).isNull();
    assertThat(classDefTree.decorators()).isNull();
  }

  @Test
  public void for_statement() {
    setRootRule(PythonGrammar.FOR_STMT);
    AstNode astNode = p.parse("for foo in bar: pass");
    PyForStatementTree pyForStatementTree = new PythonTreeMaker().forStatement(astNode);
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body()).hasSize(1);
    assertThat(pyForStatementTree.body().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody()).isEmpty();

    astNode = p.parse("for foo in bar:\n  pass\nelse:\n  pass");
    pyForStatementTree = new PythonTreeMaker().forStatement(astNode);
    assertThat(pyForStatementTree.expressions()).hasSize(1);
    assertThat(pyForStatementTree.testExpressions()).hasSize(1);
    assertThat(pyForStatementTree.body()).hasSize(1);
    assertThat(pyForStatementTree.body().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody()).hasSize(1);
    assertThat(pyForStatementTree.elseBody().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
  }

  @Test
  public void while_statement() {
    setRootRule(PythonGrammar.WHILE_STMT);
    AstNode astNode = p.parse("while foo : pass");
    PyWhileStatementTreeImpl pyForStatementTree = new PythonTreeMaker().whileStatement(astNode);
    assertThat(pyForStatementTree.condition()).isNotNull();
    assertThat(pyForStatementTree.body()).hasSize(1);
    assertThat(pyForStatementTree.body().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody()).isEmpty();

    astNode = p.parse("while foo:\n  pass\nelse:\n  pass");
    pyForStatementTree = new PythonTreeMaker().whileStatement(astNode);
    assertThat(pyForStatementTree.condition()).isNotNull();
    assertThat(pyForStatementTree.body()).hasSize(1);
    assertThat(pyForStatementTree.body().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(pyForStatementTree.elseBody()).hasSize(1);
    assertThat(pyForStatementTree.elseBody().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
  }
}
