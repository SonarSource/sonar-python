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
}
