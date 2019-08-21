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
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonTreeMakerTest extends RuleTest {

  @Test
  public void fileInputTreeOnEmptyFile() {
    AstNode astNode = p.parse("");
    PyFileInputTree pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).isEmpty();

    astNode = p.parse("pass");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyPassStatementTree.class);

    astNode = p.parse("print 'foo'");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyPrintStatementTree.class);

    astNode = p.parse("exec foo");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyExecStatementTree.class);

    astNode = p.parse("assert foo");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyAssertStatementTree.class);

    astNode = p.parse("del foo");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyDelStatementTree.class);

    astNode = p.parse("return foo");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyReturnStatementTree.class);

    astNode = p.parse("yield foo");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyYieldStatementTree.class);

    astNode = p.parse("raise foo");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyRaiseStatementTree.class);

    astNode = p.parse("break");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyBreakStatementTree.class);

    astNode = p.parse("continue");
    pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).hasSize(1);
    assertThat(pyTree.statements().get(0)).isInstanceOf(PyContinueStatementTree.class);
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
  public void funcdef_statement() {
    setRootRule(PythonGrammar.FUNCDEF);
    AstNode astNode = p.parse("def func(): pass");
    PyFunctionDefTree functionDefTree = new PythonTreeMaker().funcdefStatement(astNode);
    assertThat(functionDefTree.name()).isNotNull();
    assertThat(functionDefTree.name().name()).isEqualTo("func");
    assertThat(functionDefTree.body()).hasSize(1);
    assertThat(functionDefTree.body().get(0).is(Tree.Kind.PASS_STMT)).isTrue();
    assertThat(functionDefTree.typedArgs()).isNull();
  }
}
