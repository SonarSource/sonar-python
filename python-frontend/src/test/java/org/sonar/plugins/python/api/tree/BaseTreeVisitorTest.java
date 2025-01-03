/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.api.tree;

import com.sonar.sslr.api.AstNode;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;
import org.sonar.python.tree.PythonTreeMaker;
import org.sonar.python.tree.StatementWithSeparator;
import org.sonar.python.tree.WhileStatementImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class BaseTreeVisitorTest extends RuleTest {
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
  void if_statement() {
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
  void exec_statement() {
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
  void assert_statement() {
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
  void delete_statement() {
    setRootRule(PythonGrammar.DEL_STMT);
    AstNode astNode = p.parse("del x");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    DelStatement tree = treeMaker.delStatement(statementWithSeparator);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitDelStatement(tree);
    verify(visitor).scan(tree.expressions());
  }

  @Test
  void fundef_statement() {
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
  void fundef_with_type_params() {
    setRootRule(PythonGrammar.FUNCDEF);
    var pyFunctionDefTree = parse("def foo[A,B](): pass", treeMaker::funcDefStatement);
    var visitor = spy(FirstLastTokenVerifierVisitor.class);

    pyFunctionDefTree.accept(visitor);
    var typeParams = Objects.requireNonNull(pyFunctionDefTree.typeParams());
    verify(visitor, times(1)).visitTypeParams(typeParams);
    verify(visitor, times(2)).visitTypeParam(any());
  }

  @Test
  void fundef_with_tuple_param() {
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
  void import_statement() {
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
  void for_statement() {
    setRootRule(PythonGrammar.FOR_STMT);
    ForStatement tree = parse("for foo in bar:pass\nelse: pass", treeMaker::forStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitForStatement(tree);
    verify(visitor).visitPassStatement((PassStatement) tree.body().statements().get(0));
    verify(visitor).visitPassStatement((PassStatement) tree.elseClause().body().statements().get(0));
  }

  @Test
  void while_statement() {
    setRootRule(PythonGrammar.WHILE_STMT);
    WhileStatementImpl tree = parse("while foo:\n  pass\nelse:\n  pass", treeMaker::whileStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitWhileStatement(tree);
    verify(visitor).visitPassStatement((PassStatement) tree.body().statements().get(0));
    verify(visitor).visitPassStatement((PassStatement) tree.elseClause().body().statements().get(0));
  }

  @Test
  void try_statement() {
    setRootRule(PythonGrammar.TRY_STMT);
    TryStatement tree = parse("try: pass\nexcept Error: pass\nfinally: pass", treeMaker::tryStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitTryStatement(tree);
    verify(visitor).visitFinallyClause(tree.finallyClause());
    verify(visitor).visitExceptClause(tree.exceptClauses().get(0));
    verify(visitor).visitPassStatement((PassStatement) tree.body().statements().get(0));
  }

  @Test
  void with_statement() {
    setRootRule(PythonGrammar.WITH_STMT);
    WithStatement tree = parse("with foo as bar, qix : pass", treeMaker::withStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitWithStatement(tree);
    verify(visitor).visitWithItem(tree.withItems().get(0));
    verify(visitor).visitPassStatement((PassStatement) tree.statements().statements().get(0));
  }

  @Test
  void class_statement() {
    setRootRule(PythonGrammar.CLASSDEF);
    ClassDef tree = parse("@A\nclass clazz(Parent): pass", treeMaker::classDefStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitClassDef(tree);
    verify(visitor).visitDecorator(tree.decorators().get(0));
    verify(visitor).visitName(tree.name());
    verify(visitor).visitArgumentList(tree.args());
    verify(visitor).visitPassStatement((PassStatement) tree.body().statements().get(0));
  }

  @Test
  void class_with_type_params() {
    setRootRule(PythonGrammar.CLASSDEF);
    var classDef = parse("class clazz[A,B](): pass", treeMaker::classDefStatement);
    var visitor = spy(FirstLastTokenVerifierVisitor.class);

    classDef.accept(visitor);
    var typeParams = Objects.requireNonNull(classDef.typeParams());
    verify(visitor, times(1)).visitTypeParams(typeParams);
    verify(visitor, times(2)).visitTypeParam(any());
  }

  @Test
  void qualified_expr() {
    setRootRule(PythonGrammar.TEST);
    QualifiedExpression tree = (QualifiedExpression) parse("a.b", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitQualifiedExpression(tree);
    verify(visitor).visitName(tree.name());
  }

  @Test
  void assignement_stmt() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("a = b");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AssignmentStatement tree = treeMaker.assignment(statementWithSeparator);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitAssignmentStatement(tree);
    verify(visitor).visitExpressionList(tree.lhsExpressions().get(0));
  }

  @Test
  void assignment_expr() {
    setRootRule(PythonGrammar.NAMED_EXPR_TEST);
    AstNode astNode = p.parse("b := 42");
    AssignmentExpression assignmentExpression = (AssignmentExpression) treeMaker.expression(astNode);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    assignmentExpression.accept(visitor);
    verify(visitor).visitNumericLiteral((NumericLiteral) assignmentExpression.expression());
  }

  @Test
  void annotated_assignment() {
    setRootRule(PythonGrammar.EXPRESSION_STMT);
    AstNode astNode = p.parse("a : int = b");
    StatementWithSeparator statementWithSeparator = new StatementWithSeparator(astNode, null);
    AnnotatedAssignment tree = treeMaker.annotatedAssignment(statementWithSeparator);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitAnnotatedAssignment(tree);
    verify(visitor).visitName((Name) tree.variable());
    verify(visitor).visitTypeAnnotation(tree.annotation());
    verify(visitor).visitName((Name) tree.assignedValue());
  }

  @Test
  void lambda() {
    setRootRule(PythonGrammar.LAMBDEF);
    LambdaExpression tree = parse("lambda x : x", treeMaker::lambdaExpression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    visitor.visitLambda(tree);
    verify(visitor).visitParameterList(tree.parameters());
    verify(visitor).visitParameter(tree.parameters().nonTuple().get(0));
  }

  @Test
  void starred_expr() {
    setRootRule(PythonGrammar.STAR_EXPR);
    UnpackingExpression tree = (UnpackingExpression) parse("*a", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    tree.accept(visitor);
    verify(visitor).visitName((Name) tree.expression());
  }

  @Test
  void await_expr() {
    setRootRule(PythonGrammar.EXPR);
    AwaitExpression tree = (AwaitExpression) parse("await x", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    tree.accept(visitor);
    verify(visitor).visitName((Name) tree.expression());
  }

  @Test
  void slice_expr() {
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
  void subscription_expr() {
    setRootRule(PythonGrammar.EXPR);
    SubscriptionExpression expr = (SubscriptionExpression) parse("a[b]", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((Name) expr.object());
    verify(visitor).visitName((Name) expr.subscripts().expressions().get(0));
  }

  @Test
  void parenthesized_expr() {
    setRootRule(PythonGrammar.EXPR);
    ParenthesizedExpression expr = (ParenthesizedExpression) parse("(a)", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((Name) expr.expression());
  }

  @Test
  void tuple() {
    setRootRule(PythonGrammar.EXPR);
    Tuple expr = (Tuple) parse("(a,)", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((Name) expr.elements().get(0));
  }

  @Test
  void cond_expression() {
    setRootRule(PythonGrammar.TEST);
    ConditionalExpression expr = (ConditionalExpression) parse("1 if p else 2", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);
    verify(visitor).visitName((Name) expr.condition());
    verify(visitor).visitNumericLiteral((NumericLiteral) expr.trueExpression());
    verify(visitor).visitNumericLiteral((NumericLiteral) expr.falseExpression());
  }

  @Test
  void list_or_set_comprehension() {
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
  void dict_comprehension() {
    setRootRule(PythonGrammar.TEST);
    DictCompExpression expr = (DictCompExpression) parse("{x+1:y-1 for x,y in map}", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);

    verify(visitor).visitBinaryExpression((BinaryExpression) expr.keyExpression());
    verify(visitor).visitBinaryExpression((BinaryExpression) expr.valueExpression());
    verify(visitor).visitComprehensionFor(expr.comprehensionFor());
  }

  @Test
  void repr_expression() {
    setRootRule(PythonGrammar.ATOM);
    ReprExpression expr = (ReprExpression) parse("`1`", treeMaker::expression);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    expr.accept(visitor);

    verify(visitor).visitNumericLiteral((NumericLiteral) expr.expressionList().expressions().get(0));
  }

  @Test
  void match_stmt() {
    setRootRule(PythonGrammar.MATCH_STMT);
    MatchStatement matchStatement = parse("match command:\n" +
      "    case \"quit\" if True:\n" +
      "        ...\n" +
      "    case \"foo\" if x:=cond:\n" +
      "        ...\n", treeMaker::matchStatement);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    matchStatement.accept(visitor);

    verify(visitor).visitName(((Name) matchStatement.subjectExpression()));
    verify(visitor).visitCaseBlock(matchStatement.caseBlocks().get(0));
    verify(visitor).visitCaseBlock(matchStatement.caseBlocks().get(1));
  }

  @Test
  void case_block() {
    setRootRule(PythonGrammar.CASE_BLOCK);
    CaseBlock caseBlock = parse("case 'quit' if True: ...", treeMaker::caseBlock);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    caseBlock.accept(visitor);

    verify(visitor).visitLiteralPattern(((LiteralPattern) caseBlock.pattern()));
    verify(visitor).visitGuard(caseBlock.guard());
    verify(visitor).visitStatementList(caseBlock.body());
  }

  @Test
  void guard() {
    setRootRule(PythonGrammar.GUARD);
    Guard guard = parse("if x > 10", treeMaker::guard);
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    guard.accept(visitor);

    verify(visitor).visitBinaryExpression(((BinaryExpression) guard.condition()));
  }

  @Test
  void as_pattern() {
    setRootRule(PythonGrammar.AS_PATTERN);
    AsPattern pattern = ((AsPattern) parse("'foo' as x", treeMaker::pattern));
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    pattern.accept(visitor);

    verify(visitor).visitLiteralPattern((LiteralPattern) pattern.pattern());
    verify(visitor).visitCapturePattern(pattern.alias());
  }

  @Test
  void or_pattern() {
    setRootRule(PythonGrammar.OR_PATTERN);
    OrPattern pattern = ((OrPattern) parse("'foo' | 'bar'", treeMaker::pattern));
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    pattern.accept(visitor);

    verify(visitor).visitLiteralPattern((LiteralPattern) pattern.patterns().get(0));
    verify(visitor).visitLiteralPattern((LiteralPattern) pattern.patterns().get(1));
  }

  @Test
  void capture_pattern() {
    setRootRule(PythonGrammar.CLOSED_PATTERN);
    CapturePattern pattern = ((CapturePattern) parse("x", treeMaker::closedPattern));
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    pattern.accept(visitor);

    verify(visitor).visitName(pattern.name());
  }

  @Test
  void sequence_pattern() {
    setRootRule(PythonGrammar.CLOSED_PATTERN);
    SequencePattern pattern = ((SequencePattern) parse("[x, y]", treeMaker::closedPattern));
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    pattern.accept(visitor);

    verify(visitor).visitCapturePattern(((CapturePattern) pattern.elements().get(0)));
    verify(visitor).visitCapturePattern(((CapturePattern) pattern.elements().get(1)));
  }

  @Test
  void star_pattern() {
    setRootRule(PythonGrammar.CLOSED_PATTERN);
    SequencePattern pattern = ((SequencePattern) parse("[x, *rest]", treeMaker::closedPattern));
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    pattern.accept(visitor);

    verify(visitor).visitCapturePattern(((CapturePattern) pattern.elements().get(0)));
    StarPattern starPattern = (StarPattern) pattern.elements().get(1);
    verify(visitor).visitStarPattern(starPattern);
    verify(visitor).visitCapturePattern(((CapturePattern) starPattern.pattern()));
  }

  @Test
  void group_pattern() {
    setRootRule(PythonGrammar.CLOSED_PATTERN);
    GroupPattern pattern = ((GroupPattern) parse("(x)", treeMaker::closedPattern));
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    pattern.accept(visitor);

    verify(visitor).visitCapturePattern(((CapturePattern) pattern.pattern()));
  }

  @Test
  void class_pattern() {
    setRootRule(PythonGrammar.CLOSED_PATTERN);
    ClassPattern pattern = ((ClassPattern) parse("A(x=42)", treeMaker::closedPattern));
    FirstLastTokenVerifierVisitor visitor = spy(FirstLastTokenVerifierVisitor.class);
    pattern.accept(visitor);

    verify(visitor).visitName(((Name) pattern.targetClass()));
    KeywordPattern keywordPattern = (KeywordPattern) pattern.arguments().get(0);
    verify(visitor).visitKeywordPattern(keywordPattern);
    verify(visitor).visitName(keywordPattern.attributeName());
    verify(visitor).visitLiteralPattern(((LiteralPattern) keywordPattern.pattern()));
  }

  @Test
  void type_alias_statement() {
    setRootRule(PythonGrammar.TYPE_ALIAS_STMT);
    var node = p.parse("type A[B] = str");
    var statementWithSeparator = new StatementWithSeparator(node, null);
    var typeAliasStatement = treeMaker.typeAliasStatement(statementWithSeparator);

    var visitor = spy(FirstLastTokenVerifierVisitor.class);
    typeAliasStatement.accept(visitor);

    verify(visitor).visitTypeAliasStatement(typeAliasStatement);
    verify(visitor).visitName(typeAliasStatement.name());
    verify(visitor).visitName((Name) typeAliasStatement.expression());
  }
}
