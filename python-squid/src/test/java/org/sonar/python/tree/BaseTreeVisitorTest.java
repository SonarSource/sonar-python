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
import java.util.List;
import java.util.function.Function;
import javax.annotation.Nullable;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.tree.PyAnnotatedAssignmentTree;
import org.sonar.python.api.tree.PyAnyParameterTree;
import org.sonar.python.api.tree.PyAssertStatementTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyAwaitExpressionTree;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyComprehensionForTree;
import org.sonar.python.api.tree.PyComprehensionIfTree;
import org.sonar.python.api.tree.PyConditionalExpressionTree;
import org.sonar.python.api.tree.PyDelStatementTree;
import org.sonar.python.api.tree.PyDictCompExpressionTree;
import org.sonar.python.api.tree.PyExecStatementTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyImportFromTree;
import org.sonar.python.api.tree.PyImportNameTree;
import org.sonar.python.api.tree.PyLambdaExpressionTree;
import org.sonar.python.api.tree.PyListLiteralTree;
import org.sonar.python.api.tree.PyComprehensionExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyNumericLiteralTree;
import org.sonar.python.api.tree.PyParameterTree;
import org.sonar.python.api.tree.PyParenthesizedExpressionTree;
import org.sonar.python.api.tree.PyPassStatementTree;
import org.sonar.python.api.tree.PyPrintStatementTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.PyReprExpressionTree;
import org.sonar.python.api.tree.PyReturnStatementTree;
import org.sonar.python.api.tree.PySliceExpressionTree;
import org.sonar.python.api.tree.PySliceItemTree;
import org.sonar.python.api.tree.PyStarredExpressionTree;
import org.sonar.python.api.tree.PySubscriptionExpressionTree;
import org.sonar.python.api.tree.PyTryStatementTree;
import org.sonar.python.api.tree.PyTupleParameterTree;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.PyWithStatementTree;
import org.sonar.python.api.tree.PyYieldStatementTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.parser.RuleTest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

public class BaseTreeVisitorTest extends RuleTest {
  private final PythonTreeMaker treeMaker = new PythonTreeMaker();

  private static class FirstLastTokenVerifierVisitor extends BaseTreeVisitor {
    public FirstLastTokenVerifierVisitor() {}

    @Override
    protected void scan(@Nullable Tree tree) {
      if (tree != null) {
        assertThat(tree.firstToken()).isNotNull();
        assertThat(tree.lastToken()).isNotNull();
      }
      super.scan(tree);
    }
  }

  @Test
  public void if_statement() {
    setRootRule(PythonGrammar.IF_STMT);
    PyIfStatementTree tree = parse("if p1: print 'a'\nelif p2: return\nelse: yield", treeMaker::ifStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitIfStatement(tree);
    verify(visitor).visitIfStatement(tree);
    verify(visitor).visitIfStatement(tree.elifBranches().get(0));
    verify(visitor).visitPrintStatement((PyPrintStatementTree) tree.body().statements().get(0));
    verify(visitor).visitReturnStatement((PyReturnStatementTree) tree.elifBranches().get(0).body().statements().get(0));
    verify(visitor).visitYieldStatement((PyYieldStatementTree) tree.elseBranch().body().statements().get(0));
  }

  @Test
  public void exec_statement() {
    setRootRule(PythonGrammar.EXEC_STMT);
    PyExecStatementTree tree = parse("exec 'foo' in globals, locals", treeMaker::execStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitExecStatement(tree);
    verify(visitor).scan(tree.expression());
    verify(visitor).scan(tree.globalsExpression());
    verify(visitor).scan(tree.localsExpression());
  }

  @Test
  public void assert_statement() {
    setRootRule(PythonGrammar.ASSERT_STMT);
    PyAssertStatementTree tree = parse("assert x, y", treeMaker::assertStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitAssertStatement(tree);
    verify(visitor).scan(tree.expressions());
  }

  @Test
  public void delete_statement() {
    setRootRule(PythonGrammar.DEL_STMT);
    PyDelStatementTree tree = parse("del x", treeMaker::delStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitDelStatement(tree);
    verify(visitor).scan(tree.expressions());
  }

  @Test
  public void fundef_statement() {
    setRootRule(PythonGrammar.FUNCDEF);
    PyFunctionDefTree pyFunctionDefTree = parse("def foo(x:int): pass", treeMaker::funcDefStatement);
    PyParameterTree parameter = pyFunctionDefTree.parameters().nonTuple().get(0);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitFunctionDef(pyFunctionDefTree);
    verify(visitor).visitName(pyFunctionDefTree.name());
    verify(visitor).visitParameter(parameter);
    verify(visitor).visitTypeAnnotation(parameter.typeAnnotation());
    verify(visitor).visitPassStatement((PyPassStatementTree) pyFunctionDefTree.body().statements().get(0));
  }

  @Test
  public void fundef_with_tuple_param() {
    setRootRule(PythonGrammar.FUNCDEF);
    PyFunctionDefTree pyFunctionDefTree = parse("def foo(x, (y, z)): pass", treeMaker::funcDefStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitFunctionDef(pyFunctionDefTree);

    List<PyAnyParameterTree> parameters = pyFunctionDefTree.parameters().all();
    PyTupleParameterTree tupleParam = (PyTupleParameterTree) parameters.get(1);
    verify(visitor).visitParameter((PyParameterTree) parameters.get(0));
    verify(visitor).visitTupleParameter(tupleParam);
    verify(visitor).visitParameter((PyParameterTree) tupleParam.parameters().get(0));
    verify(visitor).visitParameter((PyParameterTree) tupleParam.parameters().get(1));
  }

  @Test
  public void import_statement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    PyImportFromTree tree = (PyImportFromTree) parse("from foo import f as g", treeMaker::importStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitImportFrom(tree);
    verify(visitor).visitAliasedName(tree.importedNames().get(0));
    verify(visitor).visitDottedName(tree.module());

    PyImportNameTree pyTree = (PyImportNameTree) parse("import f as g", treeMaker::importStatement);
    visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitImportName(pyTree);
    verify(visitor).visitAliasedName(pyTree.modules().get(0));
  }

  @Test
  public void for_statement() {
    setRootRule(PythonGrammar.FOR_STMT);
    PyForStatementTree tree = parse("for foo in bar:pass\nelse: pass", treeMaker::forStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitForStatement(tree);
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.body().statements().get(0));
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.elseBody().statements().get(0));
  }

  @Test
  public void while_statement() {
    setRootRule(PythonGrammar.WHILE_STMT);
    PyWhileStatementTreeImpl tree = parse("while foo:\n  pass\nelse:\n  pass", treeMaker::whileStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitWhileStatement(tree);
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.body().statements().get(0));
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.elseBody().statements().get(0));
  }

  @Test
  public void try_statement() {
    setRootRule(PythonGrammar.TRY_STMT);
    PyTryStatementTree tree = parse("try: pass\nexcept Error: pass\nfinally: pass", treeMaker::tryStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitTryStatement(tree);
    verify(visitor).visitFinallyClause(tree.finallyClause());
    verify(visitor).visitExceptClause(tree.exceptClauses().get(0));
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.body().statements().get(0));
  }

  @Test
  public void with_statement() {
    setRootRule(PythonGrammar.WITH_STMT);
    PyWithStatementTree tree = parse("with foo as bar, qix : pass", treeMaker::withStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitWithStatement(tree);
    verify(visitor).visitWithItem(tree.withItems().get(0));
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.statements().statements().get(0));
  }

  @Test
  public void class_statement() {
    setRootRule(PythonGrammar.CLASSDEF);
    PyClassDefTree tree = parse("class clazz(Parent): pass", treeMaker::classDefStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitClassDef(tree);
    verify(visitor).visitName(tree.name());
    verify(visitor).visitArgumentList(tree.args());
    verify(visitor).visitPassStatement((PyPassStatementTree) tree.body().statements().get(0));
  }

  @Test
  public void qualified_expr() {
    setRootRule(PythonGrammar.ATTRIBUTE_REF);
    PyQualifiedExpressionTree tree = parse("a.b", treeMaker::qualifiedExpression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitQualifiedExpression(tree);
    verify(visitor).visitName(tree.name());
  }

  @Test
  public void assignement_stmt() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    PyAssignmentStatementTree tree = parse("a = b", treeMaker::assignment);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitAssignmentStatement(tree);
    verify(visitor).visitExpressionList(tree.lhsExpressions().get(0));
  }

  @Test
  public void annotated_assignment() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    PyAnnotatedAssignmentTree tree = parse("a : int = b", treeMaker::annotatedAssignment);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitAnnotatedAssignment(tree);
    verify(visitor).visitName((PyNameTree) tree.variable());
    verify(visitor).visitName((PyNameTree) tree.annotation());
    verify(visitor).visitName((PyNameTree) tree.assignedValue());
  }

  @Test
  public void lambda() {
    setRootRule(PythonGrammar.LAMBDEF);
    PyLambdaExpressionTree tree = parse("lambda x : x", treeMaker::lambdaExpression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitLambda(tree);
    verify(visitor).visitParameterList(tree.parameters());
    verify(visitor).visitParameter(tree.parameters().nonTuple().get(0));
  }

  @Test
  public void starred_expr() {
    setRootRule(PythonGrammar.STAR_EXPR);
    PyStarredExpressionTree tree = (PyStarredExpressionTree) parse("*a", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    tree.accept(visitor);
    verify(visitor).visitName((PyNameTree) tree.expression());
  }

  @Test
  public void await_expr() {
    setRootRule(PythonGrammar.EXPR);
    PyAwaitExpressionTree tree = (PyAwaitExpressionTree) parse("await x", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    tree.accept(visitor);
    verify(visitor).visitName((PyNameTree) tree.expression());
  }

  @Test
  public void slice_expr() {
    setRootRule(PythonGrammar.EXPR);
    PySliceExpressionTree expr = (PySliceExpressionTree) parse("a[b:c:d]", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((PyNameTree) expr.object());
    verify(visitor).visitSliceList(expr.sliceList());

    PySliceItemTree slice = (PySliceItemTree) expr.sliceList().slices().get(0);
    verify(visitor).visitName((PyNameTree) slice.lowerBound());
    verify(visitor).visitName((PyNameTree) slice.upperBound());
    verify(visitor).visitName((PyNameTree) slice.stride());
  }

  @Test
  public void subscription_expr() {
    setRootRule(PythonGrammar.EXPR);
    PySubscriptionExpressionTree expr = (PySubscriptionExpressionTree) parse("a[b]", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((PyNameTree) expr.object());
    verify(visitor).visitName((PyNameTree) expr.subscripts().expressions().get(0));
  }

  @Test
  public void parenthesized_expr() {
    setRootRule(PythonGrammar.EXPR);
    PyParenthesizedExpressionTree expr = (PyParenthesizedExpressionTree) parse("(a)", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((PyNameTree) expr.expression());
  }

  @Test
  public void tuple() {
    setRootRule(PythonGrammar.EXPR);
    PyTupleTree expr = (PyTupleTree) parse("(a,)", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((PyNameTree) expr.elements().get(0));
  }

  @Test
  public void cond_expression() {
    setRootRule(PythonGrammar.TEST);
    PyConditionalExpressionTree expr = (PyConditionalExpressionTree) parse("1 if p else 2", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((PyNameTree) expr.condition());
    verify(visitor).visitNumericLiteral((PyNumericLiteralTree) expr.trueExpression());
    verify(visitor).visitNumericLiteral((PyNumericLiteralTree) expr.falseExpression());
  }

  @Test
  public void list_or_set_comprehension() {
    setRootRule(PythonGrammar.EXPR);
    PyComprehensionExpressionTree expr = (PyComprehensionExpressionTree) parse("[x+1 for x in [42, 43] if cond(x)]", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);

    verify(visitor).visitBinaryExpression((PyBinaryExpressionTree) expr.resultExpression());
    verify(visitor).visitComprehensionFor(expr.comprehensionFor());

    PyComprehensionForTree forClause = expr.comprehensionFor();
    verify(visitor).visitName((PyNameTree) forClause.loopExpression());
    verify(visitor).visitListLiteral((PyListLiteralTree) forClause.iterable());

    PyComprehensionIfTree ifClause = (PyComprehensionIfTree) forClause.nestedClause();
    verify(visitor).visitCallExpression((PyCallExpressionTree) ifClause.condition());
  }

  @Test
  public void dict_comprehension() {
    setRootRule(PythonGrammar.TEST);
    PyDictCompExpressionTree expr = (PyDictCompExpressionTree) parse("{x+1:y-1 for x,y in map}", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);

    verify(visitor).visitBinaryExpression((PyBinaryExpressionTree) expr.keyExpression());
    verify(visitor).visitBinaryExpression((PyBinaryExpressionTree) expr.valueExpression());
    verify(visitor).visitComprehensionFor(expr.comprehensionFor());
  }

  @Test
  public void repr_expression() {
    setRootRule(PythonGrammar.ATOM);
    PyReprExpressionTree expr = (PyReprExpressionTree) parse("`1`", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);

    verify(visitor).visitNumericLiteral((PyNumericLiteralTree) expr.expressionList().expressions().get(0));
  }

  private <T> T parse(String code, Function<AstNode, T> func) {
    return func.apply(p.parse(code));
  }
}
