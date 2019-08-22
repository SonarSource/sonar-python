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
import java.util.function.Function;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.tree.PyAssertStatementTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyDelStatementTree;
import org.sonar.python.api.tree.PyExecStatementTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyImportFromTree;
import org.sonar.python.api.tree.PyImportNameTree;
import org.sonar.python.api.tree.PyPassStatementTree;
import org.sonar.python.api.tree.PyPrintStatementTree;
import org.sonar.python.api.tree.PyReturnStatementTree;
import org.sonar.python.api.tree.PyTryStatementTree;
import org.sonar.python.api.tree.PyWithStatementTree;
import org.sonar.python.api.tree.PyYieldStatementTree;
import org.sonar.python.parser.RuleTest;

import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

public class BaseTreeVisitorTest extends RuleTest {
  private final PythonTreeMaker treeMaker = new PythonTreeMaker();

  @Test
  public void if_statement() {
    setRootRule(PythonGrammar.IF_STMT);
    PyIfStatementTree tree = parse("if p1: print 'a'\nelif p2: return\nelse: yield", treeMaker::ifStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitIfStatement(tree);
    verify(visitor).visitIfStatement(tree);
    verify(visitor).visitIfStatement(tree.elifBranches().get(0));
    verify(visitor).visitPrintStatement((PyPrintStatementTree) tree.body().get(0));
    verify(visitor).visitReturnStatement((PyReturnStatementTree) tree.elifBranches().get(0).body().get(0));
    verify(visitor).visitYieldStatement((PyYieldStatementTree) tree.elseBranch().body().get(0));
  }

  @Test
  public void exec_statement() {
    setRootRule(PythonGrammar.EXEC_STMT);
    PyExecStatementTree tree = parse("exec 'foo' in globals, locals", treeMaker::execStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitExecStatement(tree);
    verify(visitor).scan(tree.expression());
    verify(visitor).scan(tree.globalsExpression());
    verify(visitor).scan(tree.localsExpression());
  }

  @Test
  public void assert_statement() {
    setRootRule(PythonGrammar.ASSERT_STMT);
    PyAssertStatementTree tree = parse("assert x, y", treeMaker::assertStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitAssertStatement(tree);
    verify(visitor).scan(tree.expressions());
  }

  @Test
  public void delete_statement() {
    setRootRule(PythonGrammar.DEL_STMT);
    PyDelStatementTree tree = parse("del x", treeMaker::delStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitDelStatement(tree);
    verify(visitor).scan(tree.expressions());
  }

  @Test
  public void fundef_statement() {
    setRootRule(PythonGrammar.FUNCDEF);
    PyFunctionDefTree pyFunctionDefTree = parse("def foo(): pass", treeMaker::funcDefStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitFunctionDef(pyFunctionDefTree);
    verify(visitor).visitName(pyFunctionDefTree.name());
    verify(visitor).visitPassStatement((PyPassStatementTree) pyFunctionDefTree.body().get(0));
  }

  @Test
  public void import_statement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    PyImportFromTree tree = (PyImportFromTree) parse("from foo import f as g", treeMaker::importStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitImportFrom(tree);
    verify(visitor).visitAliasedName(tree.importedNames().get(0));
    verify(visitor).visitDottedName(tree.module());

    PyImportNameTree pyTree = (PyImportNameTree) parse("import f as g", treeMaker::importStatement);
    visitor = spy(BaseTreeVisitor.class);
    visitor.visitImportName(pyTree);
    verify(visitor).visitAliasedName(pyTree.modules().get(0));
  }

  @Test
  public void for_statement() {
    setRootRule(PythonGrammar.FOR_STMT);
    PyForStatementTree tree = parse("for foo in bar:pass\nelse: pass", treeMaker::forStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitForStatement(tree);
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.body().get(0));
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.elseBody().get(0));
  }

  @Test
  public void while_statement() {
    setRootRule(PythonGrammar.WHILE_STMT);
    PyWhileStatementTreeImpl tree = parse("while foo:\n  pass\nelse:\n  pass", treeMaker::whileStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitWhileStatement(tree);
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.body().get(0));
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.elseBody().get(0));
  }

  @Test
  public void try_statement() {
    setRootRule(PythonGrammar.TRY_STMT);
    PyTryStatementTree tree = parse("try: pass\nexcept Error: pass\nfinally: pass", treeMaker::tryStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitTryStatement(tree);
    verify(visitor).visitFinallyClause(tree.finallyClause());
    verify(visitor).visitExceptClause(tree.exceptClauses().get(0));
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.body().get(0));
  }

  @Test
  public void with_statement() {
    setRootRule(PythonGrammar.WITH_STMT);
    PyWithStatementTree tree = parse("with foo as bar, qix : pass", treeMaker::withStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitWithStatement(tree);
    verify(visitor).visitWithItem(tree.withItems().get(0));
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.statements().get(0));
  }

  @Test
  public void class_statement() {
    setRootRule(PythonGrammar.CLASSDEF);
    PyClassDefTree tree = parse("class clazz: pass", treeMaker::classDefStatement);
    BaseTreeVisitor visitor = spy(BaseTreeVisitor.class);
    visitor.visitClassDef(tree);
    verify(visitor).visitName(tree.name());
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.body().get(0));
  }

  private <T> T parse(String code, Function<AstNode, T> func) {
    return func.apply(p.parse(code));
  }
}
