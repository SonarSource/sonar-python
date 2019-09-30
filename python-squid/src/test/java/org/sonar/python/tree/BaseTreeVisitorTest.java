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
import org.sonar.python.api.tree.AnnotatedAssignment;
import org.sonar.python.api.tree.AnyParameter;
import org.sonar.python.api.tree.AssertStatement;
import org.sonar.python.api.tree.AssignmentStatement;
import org.sonar.python.api.tree.AwaitExpression;
import org.sonar.python.api.tree.BinaryExpression;
import org.sonar.python.api.tree.CallExpression;
import org.sonar.python.api.tree.ClassDef;
import org.sonar.python.api.tree.ComprehensionFor;
import org.sonar.python.api.tree.ComprehensionIf;
import org.sonar.python.api.tree.ConditionalExpression;
import org.sonar.python.api.tree.DelStatement;
import org.sonar.python.api.tree.DictCompExpression;
import org.sonar.python.api.tree.ExecStatement;
import org.sonar.python.api.tree.ForStatement;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.IfStatement;
import org.sonar.python.api.tree.ImportFrom;
import org.sonar.python.api.tree.ImportName;
import org.sonar.python.api.tree.LambdaExpression;
import org.sonar.python.api.tree.ListLiteral;
import org.sonar.python.api.tree.ComprehensionExpression;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.NumericLiteral;
import org.sonar.python.api.tree.Parameter;
import org.sonar.python.api.tree.ParenthesizedExpression;
import org.sonar.python.api.tree.PassStatement;
import org.sonar.python.api.tree.PrintStatement;
import org.sonar.python.api.tree.QualifiedExpression;
import org.sonar.python.api.tree.ReprExpression;
import org.sonar.python.api.tree.ReturnStatement;
import org.sonar.python.api.tree.SliceExpression;
import org.sonar.python.api.tree.SliceItem;
import org.sonar.python.api.tree.StarredExpression;
import org.sonar.python.api.tree.SubscriptionExpression;
import org.sonar.python.api.tree.TryStatement;
import org.sonar.python.api.tree.TupleParameter;
import org.sonar.python.api.tree.Tuple;
import org.sonar.python.api.tree.WithStatement;
import org.sonar.python.api.tree.YieldStatement;
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
    IfStatement tree = parse("if p1: print 'a'\nelif p2: return\nelse: yield", treeMaker::ifStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitIfStatement(tree);
    verify(visitor).visitIfStatement(tree);
    verify(visitor).visitIfStatement(tree.elifBranches().get(0));
    verify(visitor).visitPrintStatement((PrintStatement) tree.body().statements().get(0));
    verify(visitor).visitReturnStatement((ReturnStatement) tree.elifBranches().get(0).body().statements().get(0));
    verify(visitor).visitYieldStatement((YieldStatement) tree.elseBranch().body().statements().get(0));
  }

  @Test
  public void exec_statement() {
    setRootRule(PythonGrammar.EXEC_STMT);
    AstNode astNode = p.parse("exec 'foo' in globals, locals");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ExecStatement tree = treeMaker.execStatement(statementWithSeparator);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitExecStatement(tree);
    verify(visitor).scan(tree.expression());
    verify(visitor).scan(tree.globalsExpression());
    verify(visitor).scan(tree.localsExpression());
  }

  @Test
  public void assert_statement() {
    setRootRule(PythonGrammar.ASSERT_STMT);
    AstNode astNode = p.parse("assert x, y");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AssertStatement tree = treeMaker.assertStatement(statementWithSeparator);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitAssertStatement(tree);
    verify(visitor).scan(tree.condition());
    verify(visitor).scan(tree.message());
  }

  @Test
  public void delete_statement() {
    setRootRule(PythonGrammar.DEL_STMT);
    AstNode astNode = p.parse("del x");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    DelStatement tree = treeMaker.delStatement(statementWithSeparator);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitDelStatement(tree);
    verify(visitor).scan(tree.expressions());
  }

  @Test
  public void fundef_statement() {
    setRootRule(PythonGrammar.FUNCDEF);
    FunctionDef pyFunctionDefTree = parse("def foo(x:int): pass", treeMaker::funcDefStatement);
    Parameter parameter = pyFunctionDefTree.parameters().nonTuple().get(0);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitFunctionDef(pyFunctionDefTree);
    verify(visitor).visitName(pyFunctionDefTree.name());
    verify(visitor).visitParameter(parameter);
    verify(visitor).visitTypeAnnotation(parameter.typeAnnotation());
    verify(visitor).visitPassStatement((PassStatement) pyFunctionDefTree.body().statements().get(0));
  }

  @Test
  public void fundef_with_tuple_param() {
    setRootRule(PythonGrammar.FUNCDEF);
    FunctionDef pyFunctionDefTree = parse("def foo(x, (y, z)): pass", treeMaker::funcDefStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitFunctionDef(pyFunctionDefTree);

    List<AnyParameter> parameters = pyFunctionDefTree.parameters().all();
    TupleParameter tupleParam = (TupleParameter) parameters.get(1);
    verify(visitor).visitParameter((Parameter) parameters.get(0));
    verify(visitor).visitTupleParameter(tupleParam);
    verify(visitor).visitParameter((Parameter) tupleParam.parameters().get(0));
    verify(visitor).visitParameter((Parameter) tupleParam.parameters().get(1));
  }

  @Test
  public void import_statement() {
    setRootRule(PythonGrammar.IMPORT_STMT);
    AstNode astNode = p.parse("from foo import f as g");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    ImportFrom tree = (ImportFrom) treeMaker.importStatement(statementWithSeparator);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitImportFrom(tree);
    verify(visitor).visitAliasedName(tree.importedNames().get(0));
    verify(visitor).visitDottedName(tree.module());

    astNode = p.parse("import f as g");
    statementWithSeparator = new StatementWithSeparator(astNode, null);
    ImportName pyTree = (ImportName) treeMaker.importStatement(statementWithSeparator);
    visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitImportName(pyTree);
    verify(visitor).visitAliasedName(pyTree.modules().get(0));
  }

  @Test
  public void for_statement() {
    setRootRule(PythonGrammar.FOR_STMT);
    ForStatement tree = parse("for foo in bar:pass\nelse: pass", treeMaker::forStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitForStatement(tree);
    verify(visitor).visitPassStatement((PassStatement) tree.body().statements().get(0));
    verify(visitor).visitPassStatement((PassStatement) tree.elseBody().statements().get(0));
  }

  @Test
  public void while_statement() {
    setRootRule(PythonGrammar.WHILE_STMT);
    WhileStatementImpl tree = parse("while foo:\n  pass\nelse:\n  pass", treeMaker::whileStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitWhileStatement(tree);
    verify(visitor).visitPassStatement((PassStatement) tree.body().statements().get(0));
    verify(visitor).visitPassStatement((PassStatement) tree.elseBody().statements().get(0));
  }

  @Test
  public void try_statement() {
    setRootRule(PythonGrammar.TRY_STMT);
    TryStatement tree = parse("try: pass\nexcept Error: pass\nfinally: pass", treeMaker::tryStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitTryStatement(tree);
    verify(visitor).visitFinallyClause(tree.finallyClause());
    verify(visitor).visitExceptClause(tree.exceptClauses().get(0));
    verify(visitor).visitPassStatement((PassStatement) tree.body().statements().get(0));
  }

  @Test
  public void with_statement() {
    setRootRule(PythonGrammar.WITH_STMT);
    WithStatement tree = parse("with foo as bar, qix : pass", treeMaker::withStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitWithStatement(tree);
    verify(visitor).visitWithItem(tree.withItems().get(0));
    verify(visitor).visitPassStatement((PassStatement) tree.statements().statements().get(0));
  }

  @Test
  public void class_statement() {
    setRootRule(PythonGrammar.CLASSDEF);
    ClassDef tree = parse("class clazz(Parent): pass", treeMaker::classDefStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitClassDef(tree);
    verify(visitor).visitName(tree.name());
    verify(visitor).visitArgumentList(tree.args());
    verify(visitor).visitPassStatement((PassStatement) tree.body().statements().get(0));
  }

  @Test
  public void qualified_expr() {
    setRootRule(PythonGrammar.ATTRIBUTE_REF);
    QualifiedExpression tree = parse("a.b", treeMaker::qualifiedExpression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitQualifiedExpression(tree);
    verify(visitor).visitName(tree.name());
  }

  @Test
  public void assignement_stmt() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("a = b");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AssignmentStatement tree = treeMaker.assignment(statementWithSeparator);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitAssignmentStatement(tree);
    verify(visitor).visitExpressionList(tree.lhsExpressions().get(0));
  }

  @Test
  public void annotated_assignment() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("a : int = b");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AnnotatedAssignment tree = treeMaker.annotatedAssignment(statementWithSeparator);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitAnnotatedAssignment(tree);
    verify(visitor).visitName((Name) tree.variable());
    verify(visitor).visitName((Name) tree.annotation());
    verify(visitor).visitName((Name) tree.assignedValue());
  }

  @Test
  public void lambda() {
    setRootRule(PythonGrammar.LAMBDEF);
    LambdaExpression tree = parse("lambda x : x", treeMaker::lambdaExpression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitLambda(tree);
    verify(visitor).visitParameterList(tree.parameters());
    verify(visitor).visitParameter(tree.parameters().nonTuple().get(0));
  }

  @Test
  public void starred_expr() {
    setRootRule(PythonGrammar.STAR_EXPR);
    StarredExpression tree = (StarredExpression) parse("*a", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    tree.accept(visitor);
    verify(visitor).visitName((Name) tree.expression());
  }

  @Test
  public void await_expr() {
    setRootRule(PythonGrammar.EXPR);
    AwaitExpression tree = (AwaitExpression) parse("await x", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    tree.accept(visitor);
    verify(visitor).visitName((Name) tree.expression());
  }

  @Test
  public void slice_expr() {
    setRootRule(PythonGrammar.EXPR);
    SliceExpression expr = (SliceExpression) parse("a[b:c:d]", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((Name) expr.object());
    verify(visitor).visitSliceList(expr.sliceList());

    SliceItem slice = (SliceItem) expr.sliceList().slices().get(0);
    verify(visitor).visitName((Name) slice.lowerBound());
    verify(visitor).visitName((Name) slice.upperBound());
    verify(visitor).visitName((Name) slice.stride());
  }

  @Test
  public void subscription_expr() {
    setRootRule(PythonGrammar.EXPR);
    SubscriptionExpression expr = (SubscriptionExpression) parse("a[b]", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((Name) expr.object());
    verify(visitor).visitName((Name) expr.subscripts().expressions().get(0));
  }

  @Test
  public void parenthesized_expr() {
    setRootRule(PythonGrammar.EXPR);
    ParenthesizedExpression expr = (ParenthesizedExpression) parse("(a)", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((Name) expr.expression());
  }

  @Test
  public void tuple() {
    setRootRule(PythonGrammar.EXPR);
    Tuple expr = (Tuple) parse("(a,)", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((Name) expr.elements().get(0));
  }

  @Test
  public void cond_expression() {
    setRootRule(PythonGrammar.TEST);
    ConditionalExpression expr = (ConditionalExpression) parse("1 if p else 2", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((Name) expr.condition());
    verify(visitor).visitNumericLiteral((NumericLiteral) expr.trueExpression());
    verify(visitor).visitNumericLiteral((NumericLiteral) expr.falseExpression());
  }

  @Test
  public void list_or_set_comprehension() {
    setRootRule(PythonGrammar.EXPR);
    ComprehensionExpression expr = (ComprehensionExpression) parse("[x+1 for x in [42, 43] if cond(x)]", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);

    verify(visitor).visitBinaryExpression((BinaryExpression) expr.resultExpression());
    verify(visitor).visitComprehensionFor(expr.comprehensionFor());

    ComprehensionFor forClause = expr.comprehensionFor();
    verify(visitor).visitName((Name) forClause.loopExpression());
    verify(visitor).visitListLiteral((ListLiteral) forClause.iterable());

    ComprehensionIf ifClause = (ComprehensionIf) forClause.nestedClause();
    verify(visitor).visitCallExpression((CallExpression) ifClause.condition());
  }

  @Test
  public void dict_comprehension() {
    setRootRule(PythonGrammar.TEST);
    DictCompExpression expr = (DictCompExpression) parse("{x+1:y-1 for x,y in map}", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);

    verify(visitor).visitBinaryExpression((BinaryExpression) expr.keyExpression());
    verify(visitor).visitBinaryExpression((BinaryExpression) expr.valueExpression());
    verify(visitor).visitComprehensionFor(expr.comprehensionFor());
  }

  @Test
  public void repr_expression() {
    setRootRule(PythonGrammar.ATOM);
    ReprExpression expr = (ReprExpression) parse("`1`", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);

    verify(visitor).visitNumericLiteral((NumericLiteral) expr.expressionList().expressions().get(0));
  }

  private <T> T parse(String code, Function<AstNode, T> func) {
    return func.apply(p.parse(code));
  }
}
