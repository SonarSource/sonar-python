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
package org.sonar.python.tree;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.RecognitionException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AsPattern;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BreakStatement;
import org.sonar.plugins.python.api.tree.CapturePattern;
import org.sonar.plugins.python.api.tree.CaseBlock;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ClassPattern;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.ComprehensionClause;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.ContinueStatement;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.ExecStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FinallyClause;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FormatSpecifier;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.GlobalStatement;
import org.sonar.plugins.python.api.tree.GroupPattern;
import org.sonar.plugins.python.api.tree.Guard;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.ImportStatement;
import org.sonar.plugins.python.api.tree.KeywordPattern;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.LiteralPattern;
import org.sonar.plugins.python.api.tree.MappingPattern;
import org.sonar.plugins.python.api.tree.MatchStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NonlocalStatement;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.PassStatement;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.PrintStatement;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.SequencePattern;
import org.sonar.plugins.python.api.tree.SliceItem;
import org.sonar.plugins.python.api.tree.SliceList;
import org.sonar.plugins.python.api.tree.StarPattern;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.TypeAliasStatement;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.tree.TypeParam;
import org.sonar.plugins.python.api.tree.TypeParams;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.python.DocstringExtractor;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;

public class PythonTreeMaker {

  public FileInput fileInput(AstNode astNode) {
    List<Statement> statements = getStatements(astNode).stream().map(this::statement).toList();
    StatementListImpl statementList = statements.isEmpty() ? null : new StatementListImpl(statements);
    Token endOfFile = toPyToken(astNode.getFirstChild(GenericTokenType.EOF).getToken());
    FileInputImpl pyFileInputTree = new FileInputImpl(statementList, endOfFile, DocstringExtractor.extractDocstring(statementList));
    setParents(pyFileInputTree);
    pyFileInputTree.accept(new ExceptGroupJumpInstructionsCheck());
    return pyFileInputTree;
  }

  public static void recognitionException(int line, String message) {
    throw new RecognitionException(line, "Parse error at line " + line + ": " + message + ".");
  }

  protected Token toPyToken(@Nullable com.sonar.sslr.api.Token token) {
    if (token == null) {
      return null;
    }
    return new TokenImpl(token);
  }

  protected List<Token> toPyToken(List<com.sonar.sslr.api.Token> tokens) {
    return tokens.stream().map(this::toPyToken).toList();
  }

  public void setParents(Tree root) {
    for (Tree child : root.children()) {
      if (child != null) {
        ((PyTree) child).setParent(root);
        setParents(child);
      }
    }
  }

  protected Statement statement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();

    if (astNode.is(PythonGrammar.IF_STMT)) {
      return ifStatement(astNode);
    }
    if (astNode.is(PythonGrammar.PASS_STMT)) {
      return passStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.PRINT_STMT)) {
      return printStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.EXEC_STMT)) {
      return execStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.ASSERT_STMT)) {
      return assertStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.DEL_STMT)) {
      return delStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.RETURN_STMT)) {
      return returnStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.YIELD_STMT)) {
      return yieldStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.RAISE_STMT)) {
      return raiseStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.BREAK_STMT)) {
      return breakStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.CONTINUE_STMT)) {
      return continueStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.FUNCDEF)) {
      return funcDefStatement(astNode);
    }
    if (astNode.is(PythonGrammar.CLASSDEF)) {
      return classDefStatement(astNode);
    }
    if (astNode.is(PythonGrammar.IMPORT_STMT)) {
      return importStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.FOR_STMT)) {
      return forStatement(astNode);
    }
    if (astNode.is(PythonGrammar.WHILE_STMT)) {
      return whileStatement(astNode);
    }
    if (astNode.is(PythonGrammar.GLOBAL_STMT)) {
      return globalStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.NONLOCAL_STMT)) {
      return nonlocalStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.EXPRESSION_STMT) && astNode.hasDirectChildren(PythonGrammar.ANNASSIGN)) {
      return annotatedAssignment(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.EXPRESSION_STMT) && astNode.hasDirectChildren(PythonPunctuator.ASSIGN)) {
      return assignment(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.EXPRESSION_STMT) && astNode.hasDirectChildren(PythonGrammar.AUGASSIGN)) {
      return compoundAssignment(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.EXPRESSION_STMT)) {
      return expressionStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.TYPE_ALIAS_STMT)) {
      return typeAliasStatement(statementWithSeparator);
    }
    if (astNode.is(PythonGrammar.TRY_STMT)) {
      return tryStatement(astNode);
    }
    if (astNode.is(PythonGrammar.ASYNC_STMT) && astNode.hasDirectChildren(PythonGrammar.FOR_STMT)) {
      return forStatement(astNode);
    }
    if (astNode.is(PythonGrammar.ASYNC_STMT) && astNode.hasDirectChildren(PythonGrammar.WITH_STMT)) {
      return withStatement(astNode);
    }
    if (astNode.is(PythonGrammar.WITH_STMT)) {
      return withStatement(astNode);
    }
    if (astNode.is(PythonGrammar.MATCH_STMT)) {
      return matchStatement(astNode);
    }
    throw new IllegalStateException("Statement " + astNode.getType() + " not correctly translated to strongly typed AST");
  }

  public AnnotatedAssignment annotatedAssignment(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    AstNode annAssign = astNode.getFirstChild(PythonGrammar.ANNASSIGN);
    AstNode colonTokenNode = annAssign.getFirstChild(PythonPunctuator.COLON);
    Expression variable = exprListOrTestList(astNode.getFirstChild(PythonGrammar.TESTLIST_STAR_EXPR));
    Expression annotation = expression(annAssign.getFirstChild(PythonGrammar.TEST));
    AstNode equalTokenNode = annAssign.getFirstChild(PythonPunctuator.ASSIGN);
    Token equalToken = null;
    Expression assignedValue = null;
    if (equalTokenNode != null) {
      equalToken = toPyToken(equalTokenNode.getToken());
      assignedValue = annotatedRhs(equalTokenNode.getNextSibling());
    }
    TypeAnnotationImpl typeAnnotation = new TypeAnnotationImpl(toPyToken(colonTokenNode.getToken()), null, annotation, Tree.Kind.VARIABLE_TYPE_ANNOTATION);
    return new AnnotatedAssignmentImpl(variable, typeAnnotation, equalToken, assignedValue, separators);
  }

  private StatementList getStatementListFromSuite(AstNode suite) {
    return new StatementListImpl(getStatementsFromSuite(suite));
  }

  private List<Statement> getStatementsFromSuite(AstNode astNode) {
    if (astNode.is(PythonGrammar.SUITE)) {
      List<StatementWithSeparator> statements = getStatements(astNode);
      if (statements.isEmpty()) {
        List<StatementWithSeparator> statementsWithSeparators = getStatementsWithSeparators(astNode);
        return statementsWithSeparators.stream().map(this::statement).toList();
      }
      return statements.stream().map(this::statement)
        .toList();
    }
    return Collections.emptyList();
  }

  List<StatementWithSeparator> getStatements(AstNode astNode) {
    List<AstNode> statements = astNode.getChildren(PythonGrammar.STATEMENT);
    List<StatementWithSeparator> statementsWithSeparators = new ArrayList<>();
    for (AstNode stmt : statements) {
      if (stmt.hasDirectChildren(PythonGrammar.STMT_LIST)) {
        List<StatementWithSeparator> statementList = getStatementsWithSeparators(stmt);
        statementsWithSeparators.addAll(statementList);
      } else {
        StatementWithSeparator compoundStmt = new StatementWithSeparator(stmt.getFirstChild(PythonGrammar.COMPOUND_STMT).getFirstChild(), null);
        statementsWithSeparators.add(compoundStmt);
      }
    }
    return statementsWithSeparators;
  }

  protected List<StatementWithSeparator> getStatementsWithSeparators(AstNode stmt) {
    List<StatementWithSeparator> statementsWithSeparators = new ArrayList<>();
    AstNode stmtListNode = stmt.getFirstChild(PythonGrammar.STMT_LIST);
    AstNode newLine = stmt.getFirstChild(PythonTokenType.NEWLINE);
    List<AstNode> children = stmtListNode.getChildren();
    int nbChildren = children.size();
    for (int i = 0; i < nbChildren; i += 2) {
      AstNode current = children.get(i);
      Token separator = current.getNextSibling() == null ? null : toPyToken(current.getNextSibling().getToken());
      Token newLineForSeparator = null;
      boolean isLastStmt = nbChildren - i <= 2;
      if (isLastStmt) {
        newLineForSeparator = newLine == null ? null : toPyToken(newLine.getToken());
      }
      statementsWithSeparators.add(new StatementWithSeparator(current.getFirstChild(), new Separators(separator, newLineForSeparator)));
    }
    return statementsWithSeparators;
  }

  // Simple statements
  public PrintStatement printStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    List<Expression> expressions = expressionsFromTest(astNode);
    Separators separators = statementWithSeparator.separator();
    return new PrintStatementImpl(toPyToken(astNode.getTokens()).get(0), expressions, separators);
  }

  public ExecStatement execStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Expression expression = expression(astNode.getFirstChild(PythonGrammar.EXPR));
    List<Expression> expressions = expressionsFromTest(astNode);
    Separators separators = statementWithSeparator.separator();
    if (expressions.isEmpty()) {
      return new ExecStatementImpl(toPyToken(astNode.getTokens()).get(0), expression, separators);
    }
    Token inToken = toPyToken(astNode.getFirstChild(PythonKeyword.IN).getToken());
    Expression globalsExpression = expressions.get(0);
    Token commaToken = null;
    Expression localsExpression = null;
    if (expressions.size() == 2) {
      commaToken = toPyToken(astNode.getFirstChild(PythonPunctuator.COMMA).getToken());
      localsExpression = expressions.get(1);
    }
    return new ExecStatementImpl(toPyToken(astNode.getTokens().get(0)), expression, inToken, globalsExpression, commaToken, localsExpression, separators);
  }

  public AssertStatement assertStatement(StatementWithSeparator statementWithSeparator) {
    AstNode stmt = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    List<Expression> expressions = expressionsFromTest(stmt);
    Expression condition = expressions.get(0);
    Expression message = null;
    if (expressions.size() > 1) {
      message = expressions.get(1);
    }
    return new AssertStatementImpl(toPyToken(stmt.getTokens()).get(0), condition, message, separators);
  }

  public PassStatement passStatement(StatementWithSeparator statementWithSeparator) {
    AstNode stmt = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    return new PassStatementImpl(toPyToken(stmt.getTokens()).get(0), separators);
  }

  public DelStatement delStatement(StatementWithSeparator statementWithSeparator) {
    AstNode stmt = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    List<Expression> expressionTrees = expressionsFromExprList(stmt.getFirstChild(PythonGrammar.EXPRLIST));
    return new DelStatementImpl(toPyToken(stmt.getTokens()).get(0), expressionTrees, separators);
  }

  public ReturnStatement returnStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    AstNode testListNode = astNode.getFirstChild(PythonGrammar.TESTLIST_STAR_EXPR);
    List<Expression> expressionTrees = Collections.emptyList();
    List<Token> commas = Collections.emptyList();
    if (testListNode != null) {
      expressionTrees = expressionsFromTestListStarExpr(testListNode);
      commas = punctuators(testListNode, PythonPunctuator.COMMA);
    }
    return new ReturnStatementImpl(toPyToken(astNode.getTokens()).get(0), expressionTrees, commas, separators);
  }

  public YieldStatement yieldStatement(StatementWithSeparator statementWithSeparator) {
    AstNode stmt = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    return new YieldStatementImpl(yieldExpression(stmt.getFirstChild(PythonGrammar.YIELD_EXPR)), separators);
  }

  public YieldExpression yieldExpression(AstNode astNode) {
    Token yieldKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.YIELD).getToken());
    AstNode nodeContainingExpression = astNode;
    AstNode fromKeyword = astNode.getFirstChild(PythonKeyword.FROM);
    if (fromKeyword == null) {
      nodeContainingExpression = astNode.getFirstChild(PythonGrammar.TESTLIST_STAR_EXPR);
    }
    List<Expression> expressionTrees = Collections.emptyList();
    if (nodeContainingExpression != null) {
      expressionTrees = expressionsFromTestListStarExpr(nodeContainingExpression);
    }
    return new YieldExpressionImpl(yieldKeyword, fromKeyword == null ? null : toPyToken(fromKeyword.getToken()), expressionTrees);
  }

  public RaiseStatement raiseStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    AstNode fromKeyword = astNode.getFirstChild(PythonKeyword.FROM);
    List<AstNode> expressions = new ArrayList<>();
    AstNode fromExpression = null;
    if (fromKeyword != null) {
      expressions.add(astNode.getFirstChild(PythonGrammar.TEST));
      fromExpression = astNode.getLastChild(PythonGrammar.TEST);
    } else {
      expressions = astNode.getChildren(PythonGrammar.TEST);
    }
    List<Expression> expressionTrees = expressions.stream()
      .map(this::expression)
      .toList();
    return new RaiseStatementImpl(toPyToken(astNode.getFirstChild(PythonKeyword.RAISE).getToken()),
      expressionTrees, fromKeyword == null ? null : toPyToken(fromKeyword.getToken()), fromExpression == null ? null : expression(fromExpression), separators);
  }

  public BreakStatement breakStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    return new BreakStatementImpl(toPyToken(astNode.getToken()), separators);
  }

  public ContinueStatement continueStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    return new ContinueStatementImpl(toPyToken(astNode.getToken()), separators);
  }

  public ImportStatement importStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    AstNode importStmt = astNode.getFirstChild();
    if (importStmt.is(PythonGrammar.IMPORT_NAME)) {
      return importName(importStmt, separators);
    }
    return importFromStatement(importStmt, separators);
  }

  private ImportName importName(AstNode astNode, Separators separators) {
    Token importKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.IMPORT).getToken());
    List<AliasedName> aliasedNames = astNode
      .getFirstChild(PythonGrammar.DOTTED_AS_NAMES)
      .getChildren(PythonGrammar.DOTTED_AS_NAME).stream()
      .map(this::aliasedName)
      .toList();
    return new ImportNameImpl(importKeyword, aliasedNames, separators);
  }

  private ImportFrom importFromStatement(AstNode astNode, Separators separators) {
    Token importKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.IMPORT).getToken());
    Token fromKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.FROM).getToken());
    List<Token> dottedPrefixForModule = punctuators(astNode, PythonPunctuator.DOT);
    AstNode moduleNode = astNode.getFirstChild(PythonGrammar.DOTTED_NAME);
    DottedName moduleName = null;
    if (moduleNode != null) {
      moduleName = dottedName(moduleNode);
    }
    AstNode importAsnames = astNode.getFirstChild(PythonGrammar.IMPORT_AS_NAMES);
    List<AliasedName> aliasedImportNames = null;
    boolean isWildcardImport = true;
    if (importAsnames != null) {
      aliasedImportNames = importAsnames.getChildren(PythonGrammar.IMPORT_AS_NAME).stream()
        .map(this::aliasedName)
        .toList();
      isWildcardImport = false;
    }
    Token wildcard = null;
    if (isWildcardImport) {
      wildcard = toPyToken(astNode.getFirstChild(PythonPunctuator.MUL).getToken());
    }
    return new ImportFromImpl(fromKeyword, dottedPrefixForModule, moduleName, importKeyword, aliasedImportNames, wildcard, separators);
  }

  private AliasedName aliasedName(AstNode astNode) {
    AstNode asKeyword = astNode.getFirstChild(PythonKeyword.AS);
    DottedName dottedName;
    if (astNode.is(PythonGrammar.DOTTED_AS_NAME)) {
      dottedName = dottedName(astNode.getFirstChild(PythonGrammar.DOTTED_NAME));
    } else {
      // astNode is IMPORT_AS_NAME
      AstNode importedName = astNode.getFirstChild(PythonGrammar.NAME);
      dottedName = new DottedNameImpl(Collections.singletonList(name(importedName)));
    }
    if (asKeyword == null) {
      return new AliasedNameImpl(dottedName);
    }
    return new AliasedNameImpl(toPyToken(asKeyword.getToken()), dottedName, name(astNode.getLastChild(PythonGrammar.NAME)));
  }

  private DottedName dottedName(AstNode astNode) {
    List<Name> names = astNode
      .getChildren(PythonGrammar.NAME).stream()
      .map(this::name)
      .toList();
    return new DottedNameImpl(names);
  }

  public GlobalStatement globalStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    Token globalKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.GLOBAL).getToken());
    List<Name> variables = astNode.getChildren(PythonGrammar.NAME).stream()
      .map(this::variable)
      .toList();
    return new GlobalStatementImpl(globalKeyword, variables, separators);
  }

  public NonlocalStatement nonlocalStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    Token nonlocalKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.NONLOCAL).getToken());
    List<Name> variables = astNode.getChildren(PythonGrammar.NAME).stream()
      .map(this::variable)
      .toList();
    return new NonlocalStatementImpl(nonlocalKeyword, variables, separators);
  }

  // Compound statements
  public IfStatement ifStatement(AstNode astNode) {
    Token ifToken = toPyToken(astNode.getTokens().get(0));
    AstNode condition = astNode.getFirstChild(PythonGrammar.NAMED_EXPR_TEST);
    Token colon = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    StatementList body = getStatementListFromSuite(suite);
    AstNode elseSuite = astNode.getLastChild(PythonGrammar.SUITE);
    ElseClause elseClause = null;
    if (elseSuite.getPreviousSibling().getPreviousSibling().is(PythonKeyword.ELSE)) {
      elseClause = elseClause(elseSuite);
    }
    List<IfStatement> elifBranches = astNode.getChildren(PythonKeyword.ELIF).stream()
      .map(this::elifStatement)
      .toList();

    return new IfStatementImpl(ifToken, expression(condition), colon, suiteNewLine(suite), suiteIndent(suite), body, suiteDedent(suite), elifBranches, elseClause);
  }

  private IfStatement elifStatement(AstNode astNode) {
    Token elifToken = toPyToken(astNode.getToken());
    AstNode condition = astNode.getNextSibling();
    AstNode colon = condition.getNextSibling();
    AstNode suite = colon.getNextSibling();
    StatementList body = getStatementListFromSuite(suite);
    Token colonToken = toPyToken(colon.getToken());
    return new IfStatementImpl(elifToken, expression(condition), colonToken, suiteNewLine(suite), suiteIndent(suite), body, suiteDedent(suite));
  }

  private ElseClause elseClause(AstNode astNode) {
    Token elseToken = toPyToken(astNode.getPreviousSibling().getPreviousSibling().getToken());
    Token colon = toPyToken(astNode.getPreviousSibling().getToken());
    StatementList body = getStatementListFromSuite(astNode);
    return new ElseClauseImpl(elseToken, colon, suiteNewLine(astNode), suiteIndent(astNode), body, suiteDedent(astNode));
  }

  public FunctionDef funcDefStatement(AstNode astNode) {
    AstNode decoratorsNode = astNode.getFirstChild(PythonGrammar.DECORATORS);
    List<Decorator> decorators = Collections.emptyList();
    if (decoratorsNode != null) {
      decorators = decoratorsNode.getChildren(PythonGrammar.DECORATOR).stream()
        .map(this::decorator)
        .toList();
    }
    Name name = name(astNode.getFirstChild(PythonGrammar.FUNCNAME).getFirstChild(PythonGrammar.NAME));

    var typeParams = typeParams(astNode);

    var parameterList = parameterList(astNode);

    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    StatementList body = getStatementListFromSuite(suite);
    AstNode defNode = astNode.getFirstChild(PythonKeyword.DEF);
    Token asyncToken = null;
    AstNode defPreviousSibling = defNode.getPreviousSibling();
    if (defPreviousSibling != null && defPreviousSibling.getToken().getValue().equals("async")) {
      asyncToken = toPyToken(defPreviousSibling.getToken());
    }
    Token lPar = toPyToken(astNode.getFirstChild(PythonPunctuator.LPARENTHESIS).getToken());
    Token rPar = toPyToken(astNode.getFirstChild(PythonPunctuator.RPARENTHESIS).getToken());

    TypeAnnotation returnType = null;
    AstNode returnTypeNode = astNode.getFirstChild(PythonGrammar.FUN_RETURN_ANNOTATION);
    if (returnTypeNode != null) {
      List<AstNode> children = returnTypeNode.getChildren();
      returnType = new TypeAnnotationImpl(toPyToken(children.get(0).getToken()), toPyToken(children.get(1).getToken()), expression(children.get(2)));
    }

    Token colon = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    return new FunctionDefImpl(decorators, asyncToken, toPyToken(defNode.getToken()), name, typeParams, lPar, parameterList, rPar,
      returnType, colon, suiteNewLine(suite), suiteIndent(suite), body, suiteDedent(suite),
      isMethodDefinition(astNode), DocstringExtractor.extractDocstring(body));
  }

  private ParameterList parameterList(AstNode parent) {
    return Optional.of(parent)
      .map(n -> n.getFirstChild(PythonGrammar.TYPEDARGSLIST))
      .map(n -> {
        List<AnyParameter> arguments = n.getChildren(PythonGrammar.TFPDEF, PythonPunctuator.MUL, PythonPunctuator.DIV).stream()
          .map(this::parameter).filter(Objects::nonNull).toList();
        List<Token> commas = punctuators(n, PythonPunctuator.COMMA);
        return new ParameterListImpl(arguments, commas);
      }).orElse(null);
  }

  private TypeParams typeParams(AstNode parent) {
    return Optional.of(parent)
      .map(n -> n.getFirstChild(PythonGrammar.TYPE_PARAMS))
      .map(n -> {
        var lBracket = toPyToken(n.getFirstChild(PythonPunctuator.LBRACKET).getToken());

        var parameters = Optional.of(n.getFirstChild(PythonGrammar.TYPEDARGSLIST))
          .map(argList -> argList.getChildren(PythonGrammar.TFPDEF))
          .stream()
          .flatMap(Collection::stream)
          .map(this::typeParam)
          .toList();

        var commas = Optional.of(n.getFirstChild(PythonGrammar.TYPEDARGSLIST))
          .map(argList -> punctuators(argList, PythonPunctuator.COMMA))
          .stream()
          .flatMap(Collection::stream)
          .toList();

        var rBracket = toPyToken(n.getFirstChild(PythonPunctuator.RBRACKET).getToken());

        return new TypeParamsImpl(lBracket, parameters, commas, rBracket);
      }).orElse(null);
  }

  private TypeParam typeParam(AstNode parameter) {
    var starOrStarStar = Optional.of(parameter)
      .map(AstNode::getPreviousSibling)
      .filter(ps -> ps.is(PythonPunctuator.MUL, PythonPunctuator.MUL_MUL))
      .map(ps -> toPyToken(ps.getToken()))
      .orElse(null);

    Name name = name(parameter.getFirstChild(PythonGrammar.NAME));

    var typeAnnotation = Optional.of(parameter)
      .map(p -> p.getFirstChild(PythonGrammar.TYPE_ANNOTATION))
      .map(typeAnnotationNode -> {
        var colonNode = typeAnnotationNode.getFirstChild(PythonPunctuator.COLON);
        var starNode = typeAnnotationNode.getFirstChild(PythonPunctuator.MUL);
        var testNode = typeAnnotationNode.getFirstChild(PythonGrammar.TEST);
        var colonToken = toPyToken(colonNode.getToken());
        var starToken = Optional.ofNullable(starNode)
          .map(AstNode::getToken)
          .map(this::toPyToken)
          .orElse(null);
        var testExpression = expression(testNode);
        return new TypeAnnotationImpl(colonToken, starToken, testExpression, Tree.Kind.TYPE_PARAM_TYPE_ANNOTATION);
      }).orElse(null);

    return new TypeParamImpl(starOrStarStar, name, typeAnnotation);
  }

  private Decorator decorator(AstNode astNode) {
    Token atToken = toPyToken(astNode.getFirstChild(PythonPunctuator.AT).getToken());
    Expression expression = expression(astNode.getFirstChild(PythonGrammar.NAMED_EXPR_TEST));
    Token newLineToken = astNode.getFirstChild(PythonTokenType.NEWLINE) == null ? null : toPyToken(astNode.getFirstChild(PythonTokenType.NEWLINE).getToken());
    return new DecoratorImpl(atToken, expression, newLineToken);
  }

  private static boolean isMethodDefinition(AstNode node) {
    AstNode parent = node.getParent();
    while (parent != null && !parent.is(PythonGrammar.CLASSDEF, PythonGrammar.FUNCDEF)) {
      parent = parent.getParent();
    }
    return parent != null && parent.is(PythonGrammar.CLASSDEF);
  }

  public ClassDef classDefStatement(AstNode astNode) {
    AstNode decoratorsNode = astNode.getFirstChild(PythonGrammar.DECORATORS);
    List<Decorator> decorators = Collections.emptyList();
    if (decoratorsNode != null) {
      decorators = decoratorsNode.getChildren(PythonGrammar.DECORATOR).stream()
        .map(this::decorator)
        .toList();
    }
    Name name = name(astNode.getFirstChild(PythonGrammar.CLASSNAME).getFirstChild(PythonGrammar.NAME));
    var typeParams = typeParams(astNode);

    ArgList args = null;
    AstNode leftPar = astNode.getFirstChild(PythonPunctuator.LPARENTHESIS);
    if (leftPar != null) {
      args = argList(astNode.getFirstChild(PythonGrammar.ARGLIST));
    }
    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    StatementList body = getStatementListFromSuite(suite);
    Token classToken = toPyToken(astNode.getFirstChild(PythonKeyword.CLASS).getToken());
    AstNode rightPar = astNode.getFirstChild(PythonPunctuator.RPARENTHESIS);
    Token colon = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    return new ClassDefImpl(decorators, classToken, name, typeParams,
      leftPar != null ? toPyToken(leftPar.getToken()) : null, args, rightPar != null ? toPyToken(rightPar.getToken()) : null,
      colon, suiteNewLine(suite), suiteIndent(suite), body, suiteDedent(suite), DocstringExtractor.extractDocstring(body));
  }

  protected Name name(AstNode astNode) {
    return new NameImpl(toPyToken(astNode.getFirstChild(GenericTokenType.IDENTIFIER).getToken()), astNode.getParent().is(PythonGrammar.ATOM));
  }

  private Name variable(AstNode astNode) {
    return new NameImpl(toPyToken(astNode.getFirstChild(GenericTokenType.IDENTIFIER).getToken()), true);
  }

  public ForStatement forStatement(AstNode astNode) {
    AstNode forStatementNode = astNode;
    Token asyncToken = null;
    if (astNode.is(PythonGrammar.ASYNC_STMT)) {
      asyncToken = toPyToken(astNode.getFirstChild().getToken());
      forStatementNode = astNode.getFirstChild(PythonGrammar.FOR_STMT);
    }
    Token forKeyword = toPyToken(forStatementNode.getFirstChild(PythonKeyword.FOR).getToken());
    Token inKeyword = toPyToken(forStatementNode.getFirstChild(PythonKeyword.IN).getToken());
    Token colon = toPyToken(forStatementNode.getFirstChild(PythonPunctuator.COLON).getToken());
    AstNode exprList = forStatementNode.getFirstChild(PythonGrammar.EXPRLIST);
    List<Expression> expressions = expressionsFromExprList(exprList);
    List<Token> expressionsCommas = punctuators(exprList, PythonPunctuator.COMMA);
    AstNode starNamedExpressions = forStatementNode.getFirstChild(PythonGrammar.STAR_NAMED_EXPRESSIONS);
    List<Expression> testExpressions = starNamedExpressions
      .getChildren(PythonGrammar.STAR_NAMED_EXPRESSION).stream()
      .map(this::expression)
      .toList();
    List<Token> testExpressionsCommas = punctuators(starNamedExpressions, PythonPunctuator.COMMA);
    AstNode firstSuite = forStatementNode.getFirstChild(PythonGrammar.SUITE);
    StatementList body = getStatementListFromSuite(firstSuite);
    AstNode lastSuite = forStatementNode.getLastChild(PythonGrammar.SUITE);
    ElseClause elseClause = firstSuite == lastSuite ? null : elseClause(lastSuite);
    return new ForStatementImpl(forKeyword, expressions, expressionsCommas, inKeyword, testExpressions, testExpressionsCommas,
      colon, suiteNewLine(firstSuite), suiteIndent(firstSuite), body, suiteDedent(firstSuite), elseClause, asyncToken);
  }

  public WhileStatementImpl whileStatement(AstNode astNode) {
    Token whileKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.WHILE).getToken());
    Token colon = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    Expression condition = expression(astNode.getFirstChild(PythonGrammar.NAMED_EXPR_TEST));
    AstNode firstSuite = astNode.getFirstChild(PythonGrammar.SUITE);
    StatementList body = getStatementListFromSuite(firstSuite);
    AstNode lastSuite = astNode.getLastChild(PythonGrammar.SUITE);
    ElseClause elseClause = firstSuite == lastSuite ? null : elseClause(lastSuite);
    return new WhileStatementImpl(whileKeyword, condition, colon, suiteNewLine(firstSuite), suiteIndent(firstSuite),
      body, suiteDedent(firstSuite), elseClause);
  }

  public ExpressionStatement expressionStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();

    List<Expression> expressions = astNode.getFirstChild(PythonGrammar.TESTLIST_STAR_EXPR).getChildren(PythonGrammar.TEST, PythonGrammar.STAR_EXPR).stream()
      .map(this::expression)
      .toList();
    return new ExpressionStatementImpl(expressions, separators);
  }

  public TypeAliasStatement typeAliasStatement(StatementWithSeparator statementWithSeparator) {
    var astNode = statementWithSeparator.statement();
    var separator = statementWithSeparator.separator();
    var typeDef = toPyToken(astNode.getChildren().get(0).getToken());
    var name = name(astNode.getFirstChild(PythonGrammar.NAME));
    var typeParams = typeParams(astNode);
    var equalToken = toPyToken(astNode.getFirstChild(PythonPunctuator.ASSIGN).getToken());
    var expression = expression(astNode.getFirstChild(PythonGrammar.TEST));
    return new TypeAliasStatementImpl(typeDef, name, typeParams, equalToken, expression, separator);
  }

  public AssignmentStatement assignment(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    List<Token> assignTokens = new ArrayList<>();
    List<ExpressionList> lhsExpressions = new ArrayList<>();
    List<AstNode> assignNodes = astNode.getChildren(PythonPunctuator.ASSIGN);
    for (AstNode assignNode : assignNodes) {
      assignTokens.add(toPyToken(assignNode.getToken()));
      lhsExpressions.add(expressionList(assignNode.getPreviousSibling()));
    }
    AstNode assignedValueNode = assignNodes.get(assignNodes.size() - 1).getNextSibling();
    Expression assignedValue = annotatedRhs(assignedValueNode);
    return new AssignmentStatementImpl(assignTokens, lhsExpressions, assignedValue, separators);
  }

  protected Expression annotatedRhs(AstNode annotatedRhs) {
    var child = annotatedRhs.getFirstChild();
    if (child.is(PythonGrammar.YIELD_EXPR)) {
      return yieldExpression(child);
    }
    return exprListOrTestList(child);
  }

  public CompoundAssignmentStatement compoundAssignment(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Separators separators = statementWithSeparator.separator();
    AstNode augAssignNodes = astNode.getFirstChild(PythonGrammar.AUGASSIGN);
    Expression lhsExpression = exprListOrTestList(augAssignNodes.getPreviousSibling());
    AstNode rhsAstNode = augAssignNodes.getNextSibling();
    Expression rhsExpression;
    if (rhsAstNode.is(PythonGrammar.YIELD_EXPR)) {
      rhsExpression = yieldExpression(rhsAstNode);
    } else {
      rhsExpression = exprListOrTestList(rhsAstNode);
    }
    return new CompoundAssignmentStatementImpl(lhsExpression, toPyToken(augAssignNodes.getToken()), rhsExpression, separators);
  }

  private ExpressionList expressionList(AstNode astNode) {
    if (astNode.is(PythonGrammar.TESTLIST_STAR_EXPR, PythonGrammar.TESTLIST_COMP)) {
      List<Expression> expressions = astNode.getChildren(PythonGrammar.NAMED_EXPR_TEST, PythonGrammar.TEST, PythonGrammar.STAR_EXPR).stream()
        .map(this::expression)
        .toList();
      List<Token> commas = punctuators(astNode, PythonPunctuator.COMMA);
      return new ExpressionListImpl(expressions, commas);
    }
    return new ExpressionListImpl(Collections.singletonList(expression(astNode)), Collections.emptyList());
  }

  public TryStatement tryStatement(AstNode astNode) {
    Token tryKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.TRY).getToken());
    Token colon = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    AstNode firstSuite = astNode.getFirstChild(PythonGrammar.SUITE);
    StatementList body = getStatementListFromSuite(firstSuite);
    List<ExceptClause> exceptClauseTrees = astNode.getChildren(PythonGrammar.EXCEPT_CLAUSE).stream()
      .map(except -> {
        AstNode suite = except.getNextSibling().getNextSibling();
        return exceptClause(except, getStatementListFromSuite(suite));
      })
      .toList();
    checkExceptClauses(exceptClauseTrees);
    FinallyClause finallyClause = null;
    AstNode finallyNode = astNode.getFirstChild(PythonKeyword.FINALLY);
    if (finallyNode != null) {
      Token finallyColon = toPyToken(finallyNode.getNextSibling().getToken());
      AstNode finallySuite = finallyNode.getNextSibling().getNextSibling();
      StatementList finallyBody = getStatementListFromSuite(finallySuite);
      finallyClause = new FinallyClauseImpl(toPyToken(finallyNode.getToken()), finallyColon,
        suiteNewLine(finallySuite), suiteIndent(finallySuite), finallyBody, suiteDedent(finallySuite));
    }
    ElseClause elseClauseTree = null;
    AstNode elseNode = astNode.getFirstChild(PythonKeyword.ELSE);
    if (elseNode != null) {
      elseClauseTree = elseClause(elseNode.getNextSibling().getNextSibling());
    }
    return new TryStatementImpl(tryKeyword, colon, suiteNewLine(firstSuite), suiteIndent(firstSuite), body, suiteDedent(firstSuite),
      exceptClauseTrees, finallyClause, elseClauseTree);
  }

  public void checkExceptClauses(List<ExceptClause> excepts) {
    if (excepts.isEmpty()) {
      return;
    }

    Tree.Kind firstExceptKind = excepts.get(0).getKind();
    for (ExceptClause except : excepts) {
      if (firstExceptKind != except.getKind()) {
        recognitionException(except.exceptKeyword().line(), "Try statement cannot contain both except and except* clauses");
      }
      if (except.is(Tree.Kind.EXCEPT_GROUP_CLAUSE) && except.exception() == null) {
        recognitionException(except.exceptKeyword().line(), "except* clause must specify the type of the expected exception");
      }
    }
  }

  public WithStatement withStatement(AstNode astNode) {
    AstNode withStmtNode = astNode;
    Token asyncKeyword = null;
    if (astNode.is(PythonGrammar.ASYNC_STMT)) {
      withStmtNode = astNode.getFirstChild(PythonGrammar.WITH_STMT);
      asyncKeyword = toPyToken(astNode.getFirstChild().getToken());
    }
    List<WithItem> withItems = withItems(withStmtNode.getChildren(PythonGrammar.WITH_ITEM));
    AstNode lParens = withStmtNode.getFirstChild(PythonPunctuator.LPARENTHESIS);
    Token openParens = lParens == null ? null : toPyToken(lParens.getToken());
    List<Token> commas = punctuators(withStmtNode, PythonPunctuator.COMMA);
    AstNode suite = withStmtNode.getFirstChild(PythonGrammar.SUITE);
    Token withKeyword = toPyToken(withStmtNode.getToken());
    Token colon = toPyToken(suite.getPreviousSibling().getToken());
    AstNode rParens = withStmtNode.getFirstChild(PythonPunctuator.RPARENTHESIS);
    Token closeParens = rParens == null ? null : toPyToken(rParens.getToken());
    StatementList body = getStatementListFromSuite(suite);
    return new WithStatementImpl(withKeyword, openParens, withItems, commas, closeParens, colon, suiteNewLine(suite), suiteIndent(suite), body, suiteDedent(suite), asyncKeyword);
  }

  private List<WithItem> withItems(List<AstNode> withItems) {
    return withItems.stream().map(this::withItem).toList();
  }

  private WithItem withItem(AstNode withItem) {
    AstNode testNode = withItem.getFirstChild(PythonGrammar.TEST);
    Expression test = expression(testNode);
    AstNode asNode = testNode.getNextSibling();
    Expression expr = null;
    Token as = null;
    if (asNode != null) {
      as = toPyToken(asNode.getToken());
      expr = expression(withItem.getFirstChild(PythonGrammar.EXPR));
    }
    return new WithStatementImpl.WithItemImpl(test, as, expr);
  }

  private ExceptClause exceptClause(AstNode except, StatementList body) {
    Token colon = toPyToken(except.getNextSibling().getToken());
    AstNode suite = except.getNextSibling().getNextSibling();
    Token exceptKeyword = toPyToken(except.getFirstChild(PythonKeyword.EXCEPT).getToken());
    Token star = except.getFirstChild(PythonPunctuator.MUL) == null ? null : toPyToken(except.getFirstChild(PythonPunctuator.MUL).getToken());
    Token indent = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token newLine = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token dedent = suite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.DEDENT).getToken());
    AstNode exceptionNode = except.getFirstChild(PythonGrammar.TEST);
    if (exceptionNode == null) {
      return new ExceptClauseImpl(exceptKeyword, star, colon, newLine, indent, body, dedent);
    }
    AstNode asNode = except.getFirstChild(PythonKeyword.AS);
    AstNode commaNode = except.getFirstChild(PythonPunctuator.COMMA);
    if (asNode != null || commaNode != null) {
      Expression exceptionInstance = expression(except.getLastChild(PythonGrammar.TEST));
      Token asNodeToken = asNode != null ? toPyToken(asNode.getToken()) : null;
      Token commaNodeToken = commaNode != null ? toPyToken(commaNode.getToken()) : null;
      return new ExceptClauseImpl(exceptKeyword, star, colon, newLine, indent, body, dedent, expression(exceptionNode), asNodeToken, commaNodeToken, exceptionInstance);
    }
    return new ExceptClauseImpl(exceptKeyword, star, colon, newLine, indent, body, dedent, expression(exceptionNode));
  }

  public MatchStatement matchStatement(AstNode matchStmt) {
    Token matchKeyword = toPyToken(matchStmt.getTokens().get(0));
    AstNode subjectExpr = matchStmt.getFirstChild(PythonGrammar.SUBJECT_EXPR);
    Token colon = toPyToken(matchStmt.getFirstChild(PythonPunctuator.COLON).getToken());
    Token newLine = toPyToken(matchStmt.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token indent = toPyToken(matchStmt.getFirstChild(PythonTokenType.INDENT).getToken());
    List<CaseBlock> caseBlocks = matchStmt.getChildren(PythonGrammar.CASE_BLOCK).stream().map(this::caseBlock).toList();
    Token dedent = toPyToken(matchStmt.getFirstChild(PythonTokenType.DEDENT).getToken());
    return new MatchStatementImpl(matchKeyword, expression(subjectExpr), colon, newLine, indent, caseBlocks, dedent);
  }

  public CaseBlock caseBlock(AstNode caseBlock) {
    Token caseKeyword = toPyToken(caseBlock.getTokens().get(0));
    AstNode patternOrSequence = caseBlock.getFirstChild(PythonGrammar.PATTERNS).getFirstChild();
    Pattern pattern = patternOrSequence.is(PythonGrammar.PATTERN) ? pattern(patternOrSequence.getFirstChild()) : sequencePattern(patternOrSequence);
    Guard guard = null;
    AstNode guardNode = caseBlock.getFirstChild(PythonGrammar.GUARD);
    if (guardNode != null) {
      guard = guard(guardNode);
    }
    Token colon = toPyToken(caseBlock.getFirstChild(PythonPunctuator.COLON).getToken());
    AstNode suite = caseBlock.getFirstChild(PythonGrammar.SUITE);
    StatementList body = getStatementListFromSuite(suite);
    return new CaseBlockImpl(caseKeyword, pattern, guard, colon, suiteNewLine(suite), suiteIndent(suite), body, suiteDedent(suite));
  }

  public Guard guard(AstNode guardNode) {
    Token ifKeyword = toPyToken(guardNode.getTokens().get(0));
    Expression condition = expression(guardNode.getFirstChild(PythonGrammar.NAMED_EXPR_TEST));
    return new GuardImpl(ifKeyword, condition);
  }

  public Pattern pattern(AstNode pattern) {
    if (pattern.is(PythonGrammar.OR_PATTERN)) {
      return orPattern(pattern);
    }
    return asPattern(pattern);
  }

  private Pattern orPattern(AstNode pattern) {
    List<Token> separators = punctuators(pattern, PythonPunctuator.OR);
    if (separators.isEmpty()) {
      return closedPattern(pattern.getFirstChild(PythonGrammar.CLOSED_PATTERN));
    }
    List<Pattern> patterns = pattern.getChildren(PythonGrammar.CLOSED_PATTERN).stream()
      .map(this::closedPattern)
      .toList();
    return new OrPatternImpl(patterns, separators);
  }

  private AsPattern asPattern(AstNode asPattern) {
    Pattern pattern = orPattern(asPattern.getFirstChild(PythonGrammar.OR_PATTERN));
    Token asKeyword = toPyToken(asPattern.getFirstChild(PythonKeyword.AS).getToken());
    CapturePattern alias = new CapturePatternImpl(name(asPattern.getFirstChild(PythonGrammar.CAPTURE_PATTERN).getFirstChild()));
    return new AsPatternImpl(pattern, asKeyword, alias);
  }

  public Pattern closedPattern(AstNode closedPattern) {
    AstNode astNode = closedPattern.getFirstChild();
    if (astNode.is(PythonGrammar.LITERAL_PATTERN)) {
      return literalPattern(astNode);
    } else if (astNode.is(PythonGrammar.CAPTURE_PATTERN)) {
      return new CapturePatternImpl(name(astNode.getFirstChild()));
    } else if (astNode.is(PythonGrammar.SEQUENCE_PATTERN)) {
      return sequencePattern(astNode);
    } else if (astNode.is(PythonGrammar.GROUP_PATTERN)) {
      return groupPattern(astNode);
    } else if (astNode.is(PythonGrammar.WILDCARD_PATTERN)) {
      return wildcardPattern(astNode);
    } else if (astNode.is(PythonGrammar.CLASS_PATTERN)) {
      return classPattern(astNode);
    } else if (astNode.is(PythonGrammar.VALUE_PATTERN)) {
      return new ValuePatternImpl((QualifiedExpression) nameOrAttr(astNode.getFirstChild()));
    } else if (astNode.is(PythonGrammar.MAPPING_PATTERN)) {
      return mappingPattern(astNode);
    }
    throw new IllegalStateException(String.format("Pattern %s not recognized.", astNode.getName()));
  }

  private GroupPattern groupPattern(AstNode groupPattern) {
    Token leftPar = toPyToken(groupPattern.getFirstChild(PythonPunctuator.LPARENTHESIS).getToken());
    Pattern pattern = pattern(groupPattern.getFirstChild(PythonGrammar.PATTERN).getFirstChild());
    Token rightPar = toPyToken(groupPattern.getFirstChild(PythonPunctuator.RPARENTHESIS).getToken());
    return new GroupPatternImpl(leftPar, pattern, rightPar);
  }

  private ClassPattern classPattern(AstNode classPattern) {
    Expression nameOrAttr = nameOrAttr(classPattern.getFirstChild(PythonGrammar.NAME_OR_ATTR));
    Token leftPar = punctuators(classPattern, PythonPunctuator.LPARENTHESIS).get(0);
    List<Token> commas = new ArrayList<>();
    List<Pattern> patterns = patternArgs(classPattern.getFirstChild(PythonGrammar.PATTERN_ARGS), commas);
    checkPositionalAndKeywordArgumentsConstraint(patterns);
    Token rightPar = punctuators(classPattern, PythonPunctuator.RPARENTHESIS).get(0);
    return new ClassPatternImpl(nameOrAttr, leftPar, patterns, commas, rightPar);
  }

  private static void checkPositionalAndKeywordArgumentsConstraint(List<Pattern> patterns) {
    boolean positionalArgs = true;
    for (Pattern pattern : patterns) {
      if (pattern.is(Tree.Kind.KEYWORD_PATTERN)) {
        positionalArgs = false;
      } else if (!positionalArgs) {
        int line = pattern.firstToken().line();
        recognitionException(line, "Positional patterns follow keyword patterns");
      }
    }
  }

  private List<Pattern> patternArgs(@Nullable AstNode patternArgs, List<Token> commas) {
    if (patternArgs == null) {
      return Collections.emptyList();
    }
    commas.addAll(punctuators(patternArgs, PythonPunctuator.COMMA));
    return patternArgs.getChildren(PythonGrammar.PATTERN_ARG).stream().map(arg -> patternArg(arg.getFirstChild())).toList();
  }

  private Pattern patternArg(AstNode patternArg) {
    if (patternArg.is(PythonGrammar.KEYWORD_PATTERN)) {
      return keywordPattern(patternArg);
    }
    return pattern(patternArg.getFirstChild());
  }

  private KeywordPattern keywordPattern(AstNode keywordPattern) {
    Name name = name(keywordPattern.getFirstChild(PythonGrammar.NAME));
    Token equalToken = punctuators(keywordPattern, PythonPunctuator.ASSIGN).get(0);
    Pattern pattern = pattern(keywordPattern.getFirstChild(PythonGrammar.PATTERN).getFirstChild());
    return new KeywordPatternImpl(name, equalToken, pattern);
  }

  private Expression nameOrAttr(AstNode nameOrAttr) {
    List<Token> dots = punctuators(nameOrAttr, PythonPunctuator.DOT);
    List<AstNode> names = nameOrAttr.getChildren(PythonGrammar.NAME);
    if (dots.isEmpty()) {
      return variable(names.get(0));
    }
    Expression qualifier = variable(names.get(0));

    for (int i = 1; i < names.size(); i++) {
      Name name = name(names.get(i));
      qualifier = new QualifiedExpressionImpl(name, qualifier, dots.get(i - 1));
    }
    return qualifier;
  }

  private SequencePattern sequencePattern(AstNode sequencePattern) {
    AstNode leftDelimiter = sequencePattern.getFirstChild(PythonPunctuator.LPARENTHESIS, PythonPunctuator.LBRACKET);
    AstNode rightDelimiter = sequencePattern.getFirstChild(PythonPunctuator.RPARENTHESIS, PythonPunctuator.RBRACKET);
    List<Token> commas = new ArrayList<>();
    List<Pattern> patterns = new ArrayList<>();
    if (leftDelimiter == null) {
      // sequence patterns without neither parenthesis nor square brackets.
      // In this case there needs to be at least one comma
      addPatternsAndCommasFromSequencePattern(sequencePattern, commas, patterns);
      return new SequencePatternImpl(null, patterns, commas, null);
    }
    if (leftDelimiter.is(PythonPunctuator.LPARENTHESIS)) {
      // we need to treat differently when delimiters are parenthesis '(' ')' because there needs to be at least one comma
      // e.g. '(x)' is not a sequence pattern but a group pattern instead, while '(x,)' is a sequence pattern
      AstNode openSequencePattern = sequencePattern.getFirstChild(PythonGrammar.OPEN_SEQUENCE_PATTERN);
      if (openSequencePattern != null) {
        addPatternsAndCommasFromSequencePattern(openSequencePattern, commas, patterns);
      }
    } else {
      addPatternsAndCommasFromMaybeSequencePattern(sequencePattern.getFirstChild(PythonGrammar.MAYBE_SEQUENCE_PATTERN), patterns, commas);
    }
    return new SequencePatternImpl(toPyToken(leftDelimiter.getToken()), patterns, commas, toPyToken(rightDelimiter.getToken()));
  }

  private void addPatternsAndCommasFromSequencePattern(AstNode sequencePattern, List<Token> commas, List<Pattern> patterns) {
    commas.add(toPyToken(sequencePattern.getFirstChild(PythonPunctuator.COMMA).getToken()));
    patterns.add(maybeStarPattern(sequencePattern.getFirstChild(PythonGrammar.MAYBE_STAR_PATTERN)));
    addPatternsAndCommasFromMaybeSequencePattern(sequencePattern.getFirstChild(PythonGrammar.MAYBE_SEQUENCE_PATTERN), patterns, commas);
  }

  private void addPatternsAndCommasFromMaybeSequencePattern(@Nullable AstNode maybeSequencePattern, List<Pattern> patterns, List<Token> commas) {
    if (maybeSequencePattern == null) {
      return;
    }
    patterns.addAll(maybeSequencePattern.getChildren(PythonGrammar.MAYBE_STAR_PATTERN).stream().map(this::maybeStarPattern).toList());
    commas.addAll(punctuators(maybeSequencePattern, PythonPunctuator.COMMA));
  }

  private Pattern maybeStarPattern(AstNode maybeStarPattern) {
    AstNode astNode = maybeStarPattern.getFirstChild();
    if (astNode.is(PythonGrammar.STAR_PATTERN)) {
      return starPattern(astNode);
    }
    return pattern(astNode.getFirstChild());
  }

  private StarPattern starPattern(AstNode starPattern) {
    Token starToken = toPyToken(starPattern.getFirstChild(PythonPunctuator.MUL).getToken());
    Pattern pattern;
    AstNode capturePattern = starPattern.getFirstChild(PythonGrammar.CAPTURE_PATTERN);
    if (capturePattern != null) {
      pattern = new CapturePatternImpl(name(capturePattern.getFirstChild()));
    } else {
      pattern = wildcardPattern(starPattern.getFirstChild(PythonGrammar.WILDCARD_PATTERN));
    }
    return new StarPatternImpl(starToken, pattern);
  }

  private WildcardPatternImpl wildcardPattern(AstNode wildcardPattern) {
    return new WildcardPatternImpl(toPyToken(wildcardPattern.getFirstChild().getToken()));
  }

  private MappingPattern mappingPattern(AstNode astNode) {
    Token lCurlyBrace = toPyToken(astNode.getFirstChild(PythonPunctuator.LCURLYBRACE).getToken());
    Token rCurlyBrace = toPyToken(astNode.getLastChild(PythonPunctuator.RCURLYBRACE).getToken());
    AstNode itemsPattern = astNode.getFirstChild(PythonGrammar.ITEMS_PATTERN);
    AstNode doubleStarPattern = astNode.getFirstChild(PythonGrammar.DOUBLE_STAR_PATTERN);
    if (itemsPattern == null && doubleStarPattern == null) {
      return new MappingPatternImpl(lCurlyBrace, Collections.emptyList(), Collections.emptyList(), rCurlyBrace);
    }
    List<Token> commas = new ArrayList<>();
    List<Pattern> keyValuePatterns = new ArrayList<>();
    if (itemsPattern != null) {
      commas.addAll(punctuators(itemsPattern, PythonPunctuator.COMMA));
      List<AstNode> children = itemsPattern.getChildren();
      for (AstNode currentChild : children) {
        if (currentChild.is(PythonGrammar.KEY_VALUE_PATTERN)) {
          List<AstNode> kVChildren = currentChild.getChildren();
          Pattern keyPattern;
          AstNode keyNode = kVChildren.get(0);
          if (keyNode.is(PythonGrammar.LITERAL_PATTERN)) {
            keyPattern = literalPattern(keyNode);
          } else {
            keyPattern = new ValuePatternImpl((QualifiedExpression) nameOrAttr(keyNode.getFirstChild()));
          }
          keyValuePatterns.add(new KeyValuePatternImpl(keyPattern, toPyToken(kVChildren.get(1).getToken()), pattern(kVChildren.get(2).getFirstChild())));
        }
      }
    }
    commas.addAll(punctuators(astNode, PythonPunctuator.COMMA));
    if (doubleStarPattern != null) {
      Token doubleStarToken = toPyToken(doubleStarPattern.getFirstChild(PythonPunctuator.MUL_MUL).getToken());
      CapturePattern capturePattern = new CapturePatternImpl(name(doubleStarPattern.getFirstChild(PythonGrammar.CAPTURE_PATTERN).getFirstChild()));
      keyValuePatterns.add(new DoubleStarPatternImpl(doubleStarToken, capturePattern));
    }
    return new MappingPatternImpl(lCurlyBrace, commas, keyValuePatterns, rCurlyBrace);
  }

  private LiteralPattern literalPattern(AstNode literalPattern) {
    Tree.Kind literalKind;
    if (literalPattern.hasDirectChildren(PythonGrammar.COMPLEX_NUMBER, PythonGrammar.SIGNED_NUMBER)) {
      literalKind = Tree.Kind.NUMERIC_LITERAL_PATTERN;
    } else if (literalPattern.hasDirectChildren(PythonGrammar.STRINGS)) {
      literalKind = Tree.Kind.STRING_LITERAL_PATTERN;
    } else if (literalPattern.hasDirectChildren(PythonKeyword.NONE)) {
      literalKind = Tree.Kind.NONE_LITERAL_PATTERN;
    } else {
      literalKind = Tree.Kind.BOOLEAN_LITERAL_PATTERN;
    }
    List<Token> tokens = literalPattern.getTokens().stream().map(this::toPyToken).toList();
    return new LiteralPatternImpl(tokens, literalKind);
  }

  // expressions

  private List<Expression> expressionsFromTest(AstNode astNode) {
    return astNode.getChildren(PythonGrammar.TEST).stream().map(this::expression).toList();
  }

  private List<Expression> expressionsFromTestListStarExpr(AstNode astNode) {
    return astNode
      .getChildren(PythonGrammar.TEST, PythonGrammar.STAR_EXPR)
      .stream().map(this::expression).toList();
  }

  private List<Expression> expressionsFromExprList(AstNode firstChild) {
    return firstChild
      .getChildren(PythonGrammar.EXPR, PythonGrammar.STAR_EXPR)
      .stream().map(this::expression).toList();
  }

  private Expression exprListOrTestList(AstNode exprListOrTestList) {
    List<Expression> expressions = exprListOrTestList
      .getChildren(PythonGrammar.EXPR, PythonGrammar.STAR_EXPR, PythonGrammar.TEST).stream()
      .map(this::expression)
      .toList();
    List<AstNode> commas = exprListOrTestList.getChildren(PythonPunctuator.COMMA);
    if (commas.isEmpty()) {
      return expressions.get(0);
    }
    List<Token> commaTokens = toPyToken(commas.stream().map(AstNode::getToken).toList());
    return new TupleImpl(null, expressions, commaTokens, null);
  }

  public Expression expression(AstNode astNode) {
    if (astNode.is(PythonGrammar.ATOM) && astNode.getFirstChild().is(PythonPunctuator.LBRACKET)) {
      return listLiteral(astNode);
    }
    if (astNode.is(PythonGrammar.ATOM) && astNode.getFirstChild().is(PythonPunctuator.LPARENTHESIS)) {
      return parenthesized(astNode);
    }
    if (astNode.is(PythonGrammar.ATOM) && astNode.getFirstChild().is(PythonPunctuator.LCURLYBRACE)) {
      return dictOrSetLiteral(astNode);
    }
    if (astNode.is(PythonGrammar.ATOM) && astNode.getFirstChild().is(PythonPunctuator.BACKTICK)) {
      return repr(astNode);
    }
    if (astNode.is(PythonGrammar.ATOM) && astNode.getFirstChild().is(PythonGrammar.STRINGS)) {
      return stringLiterals(astNode.getFirstChild());
    }
    if (astNode.is(PythonGrammar.ATOM) && astNode.getChildren().size() == 1) {
      return expression(astNode.getFirstChild());
    }
    if (astNode.is(PythonGrammar.TEST) && astNode.hasDirectChildren(PythonKeyword.IF)) {
      return conditionalExpression(astNode);
    }
    if (astNode.is(PythonTokenType.NUMBER)) {
      return numericLiteral(astNode);
    }
    if (astNode.is(PythonGrammar.YIELD_EXPR)) {
      return yieldExpression(astNode);
    }
    if (astNode.is(PythonGrammar.NAME)) {
      return name(astNode);
    }
    if (astNode.is(PythonGrammar.NAMED_EXPR_TEST) && astNode.hasDirectChildren(PythonPunctuator.WALRUS_OPERATOR)) {
      return assignmentExpression(astNode);
    }
    if (astNode.is(PythonGrammar.EXPR, PythonGrammar.NAMED_EXPR_TEST, PythonGrammar.TEST, PythonGrammar.TEST_NOCOND)) {
      if (astNode.getChildren().size() == 1) {
        return expression(astNode.getFirstChild());
      } else {
        return binaryExpression(astNode);
      }
    }
    if (astNode.is(
      PythonGrammar.A_EXPR, PythonGrammar.M_EXPR, PythonGrammar.SHIFT_EXPR,
      PythonGrammar.AND_EXPR, PythonGrammar.OR_EXPR, PythonGrammar.XOR_EXPR,
      PythonGrammar.AND_TEST, PythonGrammar.OR_TEST,
      PythonGrammar.COMPARISON)) {
      return binaryExpression(astNode);
    }
    if (astNode.is(PythonGrammar.POWER)) {
      return powerExpression(astNode);
    }
    if (astNode.is(PythonGrammar.LAMBDEF, PythonGrammar.LAMBDEF_NOCOND)) {
      return lambdaExpression(astNode);
    }
    if (astNode.is(PythonGrammar.FACTOR, PythonGrammar.NOT_TEST)) {
      return new UnaryExpressionImpl(toPyToken(astNode.getFirstChild().getToken()), expression(astNode.getLastChild()));
    }
    if (astNode.is(PythonGrammar.STAR_EXPR)) {
      return new UnpackingExpressionImpl(toPyToken(astNode.getToken()), expression(astNode.getLastChild()));
    }
    if (astNode.is(PythonKeyword.NONE)) {
      return new NoneExpressionImpl(toPyToken(astNode.getToken()));
    }
    if (astNode.is(PythonGrammar.ELLIPSIS)) {
      return new EllipsisExpressionImpl(toPyToken(astNode.getTokens()));
    }
    if (astNode.is(PythonGrammar.TESTLIST_STAR_EXPR)) {
      return exprListOrTestList(astNode);
    }
    if (astNode.is(PythonGrammar.STAR_NAMED_EXPRESSIONS)) {
      return starNamedExpressions(astNode);
    }
    if (astNode.is(PythonGrammar.SUBJECT_EXPR, PythonGrammar.STAR_NAMED_EXPRESSION)) {
      return expression(astNode.getFirstChild());
    }
    if (astNode.is(PythonGrammar.ANNOTATED_RHS)) {
      return annotatedRhs(astNode);
    }
    throw new IllegalStateException("Expression " + astNode.getType() + " not correctly translated to strongly typed AST");
  }

  private Expression starNamedExpressions(AstNode astNode) {
    List<Expression> expressions = astNode
      .getChildren(PythonGrammar.STAR_NAMED_EXPRESSION).stream()
      .map(this::expression)
      .toList();
    List<AstNode> commas = astNode.getChildren(PythonPunctuator.COMMA);
    if (!commas.isEmpty()) {
      List<Token> commaTokens = toPyToken(commas.stream().map(AstNode::getToken).toList());
      return new TupleImpl(null, expressions, commaTokens, null);
    }
    return expressions.get(0);
  }

  private Expression assignmentExpression(AstNode astNode) {
    AstNode nameNode = astNode.getFirstChild(PythonGrammar.TEST);
    Expression nameExpression = expression(nameNode);
    if (!nameExpression.is(Tree.Kind.NAME)) {
      int line = nameNode.getTokenLine();
      recognitionException(line, "The left-hand side of an assignment expression must be a name");
    }
    Name name = (Name) nameExpression;
    AstNode operatorNode = astNode.getFirstChild(PythonPunctuator.WALRUS_OPERATOR);
    Token operatorToken = toPyToken(operatorNode.getToken());
    Expression expression = expression(astNode.getLastChild(PythonGrammar.TEST));
    return new AssignmentExpressionImpl(name, operatorToken, expression);
  }

  private Expression repr(AstNode astNode) {
    Token openingBacktick = toPyToken(astNode.getFirstChild(PythonPunctuator.BACKTICK).getToken());
    Token closingBacktick = toPyToken(astNode.getLastChild(PythonPunctuator.BACKTICK).getToken());
    List<Expression> expressions = astNode.getChildren(PythonGrammar.TEST).stream().map(this::expression).toList();
    List<Token> commas = punctuators(astNode, PythonPunctuator.COMMA);
    ExpressionList expressionListTree = new ExpressionListImpl(expressions, commas);
    return new ReprExpressionImpl(openingBacktick, expressionListTree, closingBacktick);
  }

  private List<Token> punctuators(AstNode astNode, PythonPunctuator punctuator) {
    return toPyToken(astNode.getChildren(punctuator).stream().map(AstNode::getToken).toList());
  }

  private Expression dictOrSetLiteral(AstNode astNode) {
    Token lCurlyBrace = toPyToken(astNode.getFirstChild(PythonPunctuator.LCURLYBRACE).getToken());
    Token rCurlyBrace = toPyToken(astNode.getLastChild(PythonPunctuator.RCURLYBRACE).getToken());
    AstNode dictOrSetMaker = astNode.getFirstChild(PythonGrammar.DICTORSETMAKER);
    if (dictOrSetMaker == null) {
      return new DictionaryLiteralImpl(lCurlyBrace, Collections.emptyList(), Collections.emptyList(), rCurlyBrace);
    }
    AstNode compForNode = dictOrSetMaker.getFirstChild(PythonGrammar.COMP_FOR);
    if (compForNode != null) {
      ComprehensionFor compFor = compFor(compForNode);
      AstNode colon = dictOrSetMaker.getFirstChild(PythonPunctuator.COLON);
      if (colon != null) {
        Expression keyExpression = expression(dictOrSetMaker.getFirstChild(PythonGrammar.TEST));
        Expression valueExpression = expression(dictOrSetMaker.getLastChild(PythonGrammar.TEST));
        return new DictCompExpressionImpl(lCurlyBrace, keyExpression, toPyToken(colon.getToken()), valueExpression, compFor, rCurlyBrace);
      } else {
        Expression resultExpression = expression(dictOrSetMaker.getFirstChild(PythonGrammar.STAR_NAMED_EXPRESSION));
        return new ComprehensionExpressionImpl(Tree.Kind.SET_COMPREHENSION, lCurlyBrace, resultExpression, compFor, rCurlyBrace);
      }
    }
    List<Token> commas = punctuators(dictOrSetMaker, PythonPunctuator.COMMA);
    if (dictOrSetMaker.hasDirectChildren(PythonPunctuator.COLON) || dictOrSetMaker.hasDirectChildren(PythonPunctuator.MUL_MUL)) {
      List<DictionaryLiteralElement> dictionaryLiteralElements = new ArrayList<>();
      List<AstNode> children = dictOrSetMaker.getChildren();
      int index = 0;
      while (index < children.size()) {
        AstNode currentChild = children.get(index);
        if (currentChild.is(PythonPunctuator.MUL_MUL)) {
          dictionaryLiteralElements.add(new UnpackingExpressionImpl(toPyToken(currentChild.getToken()), expression(children.get(index + 1))));
          index += 3;
        } else {
          dictionaryLiteralElements.add(new KeyValuePairImpl(expression(currentChild), toPyToken(children.get(index + 1).getToken()), expression(children.get(index + 2))));
          index += 4;
        }
      }
      return new DictionaryLiteralImpl(lCurlyBrace, commas, dictionaryLiteralElements, rCurlyBrace);
    }
    List<Expression> expressions = dictOrSetMaker.getChildren(PythonGrammar.STAR_NAMED_EXPRESSION).stream().map(this::expression).toList();
    return new SetLiteralImpl(lCurlyBrace, expressions, commas, rCurlyBrace);
  }

  private Expression parenthesized(AstNode atom) {
    Token lPar = toPyToken(atom.getFirstChild().getToken());
    Token rPar = toPyToken(atom.getLastChild().getToken());

    AstNode yieldNode = atom.getFirstChild(PythonGrammar.YIELD_EXPR);
    if (yieldNode != null) {
      return new ParenthesizedExpressionImpl(lPar, expression(yieldNode), rPar);
    }

    AstNode testListComp = atom.getFirstChild(PythonGrammar.TESTLIST_COMP);
    if (testListComp == null) {
      return new TupleImpl(lPar, Collections.emptyList(), Collections.emptyList(), rPar);
    }

    AstNode compFor = testListComp.getFirstChild(PythonGrammar.COMP_FOR);
    if (compFor != null) {
      return new ComprehensionExpressionImpl(Tree.Kind.GENERATOR_EXPR, lPar, expression(testListComp.getFirstChild()), compFor(compFor), rPar);
    }
    ExpressionList expressionList = expressionList(testListComp);
    List<AstNode> commas = testListComp.getChildren(PythonPunctuator.COMMA);
    if (commas.isEmpty()) {
      Expression expression = expressionList.expressions().get(0);
      return new ParenthesizedExpressionImpl(lPar, expression, rPar);
    }

    List<Token> commaTokens = toPyToken(commas.stream().map(AstNode::getToken).toList());
    return new TupleImpl(lPar, expressionList.expressions(), commaTokens, rPar);
  }

  private ConditionalExpression conditionalExpression(AstNode astNode) {
    List<AstNode> children = astNode.getChildren();
    Expression trueExpression = expression(children.get(0));
    Token ifToken = toPyToken(astNode.getFirstChild(PythonKeyword.IF).getToken());
    Expression condition = expression(children.get(2));
    Token elseToken = toPyToken(astNode.getFirstChild(PythonKeyword.ELSE).getToken());
    Expression falseExpression = expression(children.get(4));
    return new ConditionalExpressionImpl(trueExpression, ifToken, condition, elseToken, falseExpression);
  }

  private Expression powerExpression(AstNode astNode) {
    Expression expr = expression(astNode.getFirstChild(PythonGrammar.ATOM));
    for (AstNode trailer : astNode.getChildren(PythonGrammar.TRAILER)) {
      expr = withTrailer(expr, trailer);
    }
    if (astNode.getFirstChild().is(GenericTokenType.IDENTIFIER)) {
      expr = new AwaitExpressionImpl(toPyToken(astNode.getFirstChild().getToken()), expr);
    }
    AstNode powerOperator = astNode.getFirstChild(PythonPunctuator.MUL_MUL);
    if (powerOperator != null) {
      expr = new BinaryExpressionImpl(expr, toPyToken(powerOperator.getToken()), expression(powerOperator.getNextSibling()));
    }
    return expr;
  }

  private Expression withTrailer(Expression expr, AstNode trailer) {
    AstNode firstChild = trailer.getFirstChild();

    if (firstChild.is(PythonPunctuator.LPARENTHESIS)) {
      AstNode argListNode = trailer.getFirstChild(PythonGrammar.ARGLIST);
      ArgList argumentList = argList(argListNode);
      if (argumentList != null) {
        checkGeneratorExpressionInArgument(argumentList.arguments());
      }
      Token leftPar = toPyToken(firstChild.getToken());
      Token rightPar = toPyToken(trailer.getFirstChild(PythonPunctuator.RPARENTHESIS).getToken());
      return new CallExpressionImpl(expr, argumentList, leftPar, rightPar);

    } else if (firstChild.is(PythonPunctuator.LBRACKET)) {
      Token leftBracket = toPyToken(trailer.getFirstChild(PythonPunctuator.LBRACKET).getToken());
      Token rightBracket = toPyToken(trailer.getFirstChild(PythonPunctuator.RBRACKET).getToken());
      return subscriptionOrSlicing(expr, leftBracket, trailer.getFirstChild(PythonGrammar.SUBSCRIPTLIST), rightBracket);

    } else {
      Name name = name(trailer.getFirstChild(PythonGrammar.NAME));
      return new QualifiedExpressionImpl(name, expr, toPyToken(trailer.getFirstChild(PythonPunctuator.DOT).getToken()));
    }
  }

  private Expression subscriptionOrSlicing(Expression expr, Token leftBracket, AstNode subscriptList, Token rightBracket) {
    List<Tree> slices = new ArrayList<>();
    for (AstNode subscript : subscriptList.getChildren(PythonGrammar.SUBSCRIPT)) {
      AstNode colon = subscript.getFirstChild(PythonPunctuator.COLON);
      var slice = colon == null ? expression(subscript.getFirstChild(PythonGrammar.NAMED_EXPR_TEST, PythonGrammar.STAR_EXPR))
        : sliceItem(subscript);
      slices.add(slice);
    }

    // https://docs.python.org/3/reference/expressions.html#slicings
    // "There is ambiguity in the formal syntax here"
    // "a subscription takes priority over the interpretation as a slicing (this is the case if the slice list contains no proper slice)"
    if (slices.stream().anyMatch(s -> Tree.Kind.SLICE_ITEM.equals(s.getKind()))) {
      List<Token> separators = punctuators(subscriptList, PythonPunctuator.COMMA);
      SliceList sliceList = new SliceListImpl(slices, separators);
      return new SliceExpressionImpl(expr, leftBracket, sliceList, rightBracket);

    } else {
      List<Expression> expressions = slices.stream().map(Expression.class::cast).toList();
      List<Token> commas = punctuators(subscriptList, PythonPunctuator.COMMA);
      ExpressionList subscripts = new ExpressionListImpl(expressions, commas);
      return new SubscriptionExpressionImpl(expr, leftBracket, subscripts, rightBracket);
    }
  }

  SliceItem sliceItem(AstNode subscript) {
    AstNode boundSeparator = subscript.getFirstChild(PythonPunctuator.COLON);
    Expression lowerBound = sliceBound(boundSeparator.getPreviousSibling());
    Expression upperBound = sliceBound(boundSeparator.getNextSibling());
    AstNode strideNode = subscript.getFirstChild(PythonGrammar.SLICEOP);
    Token strideSeparator = strideNode == null ? null : toPyToken(strideNode.getToken());
    Expression stride = null;
    if (strideNode != null && strideNode.hasDirectChildren(PythonGrammar.TEST)) {
      stride = expression(strideNode.getLastChild());
    }
    return new SliceItemImpl(lowerBound, toPyToken(boundSeparator.getToken()), upperBound, strideSeparator, stride);
  }

  @CheckForNull
  private Expression sliceBound(@Nullable AstNode node) {
    if (node == null || !node.is(PythonGrammar.TEST)) {
      return null;
    }
    return expression(node);
  }

  private Expression listLiteral(AstNode astNode) {
    Token leftBracket = toPyToken(astNode.getFirstChild(PythonPunctuator.LBRACKET).getToken());
    Token rightBracket = toPyToken(astNode.getFirstChild(PythonPunctuator.RBRACKET).getToken());

    ExpressionList elements;
    AstNode testListComp = astNode.getFirstChild(PythonGrammar.TESTLIST_COMP);
    if (testListComp != null) {
      AstNode compForNode = testListComp.getFirstChild(PythonGrammar.COMP_FOR);
      if (compForNode != null) {
        Expression resultExpression = expression(testListComp.getFirstChild(PythonGrammar.NAMED_EXPR_TEST, PythonGrammar.STAR_EXPR));
        return new ComprehensionExpressionImpl(Tree.Kind.LIST_COMPREHENSION, leftBracket, resultExpression, compFor(compForNode), rightBracket);
      }
      elements = expressionList(testListComp);
    } else {
      elements = new ExpressionListImpl(Collections.emptyList(), Collections.emptyList());
    }
    return new ListLiteralImpl(leftBracket, elements, rightBracket);
  }

  private ComprehensionFor compFor(AstNode compFor) {
    Expression expression = exprListOrTestList(compFor.getFirstChild(PythonGrammar.EXPRLIST));
    AstNode forSSLRToken = compFor.getFirstChild(PythonKeyword.FOR);
    Token asyncToken = null;
    AstNode previousSibling = forSSLRToken.getPreviousSibling();
    if (previousSibling != null) {
      // previous sibling can only be "async"
      asyncToken = toPyToken(previousSibling.getToken());
    }
    Token forToken = toPyToken(forSSLRToken.getToken());
    Token inToken = toPyToken(compFor.getFirstChild(PythonKeyword.IN).getToken());
    Expression iterable = exprListOrTestList(compFor.getFirstChild(PythonGrammar.TESTLIST));
    ComprehensionClause nested = compClause(compFor.getFirstChild(PythonGrammar.COMP_ITER));
    return new ComprehensionForImpl(asyncToken, forToken, expression, inToken, iterable, nested);
  }

  @CheckForNull
  private ComprehensionClause compClause(@Nullable AstNode node) {
    if (node == null) {
      return null;
    }
    AstNode child = node.getFirstChild();
    if (child.is(PythonGrammar.COMP_FOR)) {
      return compFor(child);
    } else {
      Expression condition = expression(child.getFirstChild(PythonGrammar.TEST_NOCOND));
      ComprehensionClause nestedClause = compClause(child.getFirstChild(PythonGrammar.COMP_ITER));
      Token ifToken = toPyToken(child.getFirstChild(PythonKeyword.IF).getToken());
      return new ComprehensionIfImpl(ifToken, condition, nestedClause);
    }
  }

  @CheckForNull
  private ArgList argList(@Nullable AstNode argList) {
    if (argList != null) {
      List<Argument> arguments = argList.getChildren(PythonGrammar.ARGUMENT).stream()
        .map(this::argument)
        .toList();
      List<Token> commas = punctuators(argList, PythonPunctuator.COMMA);
      return new ArgListImpl(arguments, commas);
    }
    return null;
  }

  /*
   * Post Condition on Generator Expression: parentheses can be omitted on calls with only one argument.
   * https://docs.python.org/3/reference/expressions.html#grammar-token-generator-expression
   */
  private static void checkGeneratorExpressionInArgument(List<Argument> arguments) {
    List<Argument> nonParenthesizedGeneratorExpressions = arguments.stream()
      .filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
      .map(RegularArgument.class::cast)
      .filter(arg -> arg.expression().is(Tree.Kind.GENERATOR_EXPR) && !arg.expression().firstToken().value().equals("("))
      .collect(Collectors.toList());
    if (!nonParenthesizedGeneratorExpressions.isEmpty() && arguments.size() > 1) {
      int line = nonParenthesizedGeneratorExpressions.get(0).firstToken().line();
      recognitionException(line, "Generator expression must be parenthesized if not sole argument");
    }
  }

  public Argument argument(AstNode astNode) {
    AstNode compFor = astNode.getFirstChild(PythonGrammar.COMP_FOR);
    if (compFor != null) {
      Expression expression = expression(astNode.getFirstChild());
      ComprehensionExpression comprehension = new ComprehensionExpressionImpl(Tree.Kind.GENERATOR_EXPR, null, expression, compFor(compFor), null);
      return new RegularArgumentImpl(comprehension);
    }
    AstNode walrusOperator = astNode.getFirstChild(PythonPunctuator.WALRUS_OPERATOR);
    if (walrusOperator != null) {
      AssignmentExpression assignmentExpression = (AssignmentExpression) assignmentExpression(astNode);
      return new RegularArgumentImpl(assignmentExpression);
    }
    AstNode assign = astNode.getFirstChild(PythonPunctuator.ASSIGN);
    Token star = astNode.getFirstChild(PythonPunctuator.MUL) == null ? null : toPyToken(astNode.getFirstChild(PythonPunctuator.MUL).getToken());
    if (star == null) {
      star = astNode.getFirstChild(PythonPunctuator.MUL_MUL) == null ? null : toPyToken(astNode.getFirstChild(PythonPunctuator.MUL_MUL).getToken());
    }
    Expression arg = expression(astNode.getLastChild(PythonGrammar.TEST));
    if (assign != null) {
      // Keyword in argument list must be an identifier.
      AstNode nameNode = astNode.getFirstChild(PythonGrammar.TEST).getFirstChild(PythonGrammar.ATOM).getFirstChild(PythonGrammar.NAME);
      return new RegularArgumentImpl(name(nameNode), toPyToken(assign.getToken()), arg);
    }
    return star == null ? new RegularArgumentImpl(arg) : new UnpackingExpressionImpl(star, arg);
  }

  private Expression binaryExpression(AstNode astNode) {
    List<AstNode> children = astNode.getChildren();
    Expression result = expression(children.get(0));
    for (int i = 1; i < astNode.getNumberOfChildren(); i += 2) {
      AstNode operator = children.get(i);
      Expression rightOperand = expression(operator.getNextSibling());
      AstNode not = operator.getFirstChild(PythonKeyword.NOT);
      Token notToken = not == null ? null : toPyToken(not.getToken());
      if (PythonKeyword.IN.equals(operator.getLastToken().getType())) {
        result = new InExpressionImpl(result, notToken, toPyToken(operator.getLastToken()), rightOperand);
      } else if (PythonKeyword.IS.equals(operator.getToken().getType())) {
        result = new IsExpressionImpl(result, toPyToken(operator.getToken()), notToken, rightOperand);
      } else {
        result = new BinaryExpressionImpl(result, toPyToken(operator.getToken()), rightOperand);
      }
    }
    return result;
  }

  public LambdaExpression lambdaExpression(AstNode astNode) {
    Token lambdaKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.LAMBDA).getToken());
    Token colonToken = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    Expression body = expression(astNode.getFirstChild(PythonGrammar.TEST, PythonGrammar.TEST_NOCOND));
    AstNode varArgsListNode = astNode.getFirstChild(PythonGrammar.VARARGSLIST);
    ParameterList argListTree = null;
    if (varArgsListNode != null) {
      List<AnyParameter> parameters = varArgsListNode.getChildren(PythonGrammar.FPDEF, PythonGrammar.NAME, PythonPunctuator.MUL, PythonPunctuator.DIV).stream()
        .map(this::parameter).filter(Objects::nonNull).toList();
      List<Token> commas = punctuators(varArgsListNode, PythonPunctuator.COMMA);
      argListTree = new ParameterListImpl(parameters, commas);
    }

    return new LambdaExpressionImpl(lambdaKeyword, colonToken, body, argListTree);
  }

  private AnyParameter parameter(AstNode parameter) {
    if (parameter.is(PythonPunctuator.DIV)) {
      return new ParameterImpl(toPyToken(parameter.getToken()));
    }
    if (parameter.is(PythonPunctuator.MUL)) {
      if (parameter.getNextSibling() == null || parameter.getNextSibling().is(PythonPunctuator.COMMA)) {
        return new ParameterImpl(toPyToken(parameter.getToken()));
      }
      return null;
    }
    AstNode prevSibling = parameter.getPreviousSibling();

    if (parameter.is(PythonGrammar.NAME)) {
      return new ParameterImpl(toPyToken(prevSibling.getToken()), name(parameter), null, null, null);
    }

    // parameter is FPDEF or TFPDEF

    AstNode paramList = parameter.getFirstChild(PythonGrammar.TFPLIST, PythonGrammar.FPLIST);
    // Python 2 only, PEP 3113: Tuple parameter unpacking removed
    if (paramList != null) {
      List<AnyParameter> params = paramList.getChildren(PythonGrammar.TFPDEF, PythonGrammar.FPDEF).stream()
        .map(this::parameter)
        .toList();
      List<Token> commas = punctuators(paramList, PythonPunctuator.COMMA);
      return new TupleParameterImpl(toPyToken(parameter.getFirstChild(PythonPunctuator.LPARENTHESIS).getToken()),
        params, commas,
        toPyToken(parameter.getFirstChild(PythonPunctuator.RPARENTHESIS).getToken()));
    }

    Token starOrStarStar = null;
    if (prevSibling != null && prevSibling.is(PythonPunctuator.MUL, PythonPunctuator.MUL_MUL)) {
      starOrStarStar = toPyToken(prevSibling.getToken());
    }

    Name name = name(parameter.getFirstChild(PythonGrammar.NAME));

    AstNode nextSibling = parameter.getNextSibling();
    Token assignToken = null;
    Expression defaultValue = null;
    if (nextSibling != null && nextSibling.is(PythonPunctuator.ASSIGN)) {
      assignToken = toPyToken(nextSibling.getToken());
      defaultValue = expression(nextSibling.getNextSibling());
    }

    TypeAnnotation typeAnnotation = null;
    AstNode typeAnnotationNode = parameter.getFirstChild(PythonGrammar.TYPE_ANNOTATION);
    if (typeAnnotationNode != null) {
      var testNode = typeAnnotationNode.getFirstChild(PythonGrammar.TEST);
      Token colonToken = toPyToken(typeAnnotationNode.getFirstChild(PythonPunctuator.COLON).getToken());
      var starToken = Optional.ofNullable(typeAnnotationNode.getFirstChild(PythonPunctuator.MUL))
        .map(AstNode::getToken)
        .map(this::toPyToken)
        .orElse(null);
      typeAnnotation = new TypeAnnotationImpl(colonToken, starToken, expression(testNode), Tree.Kind.PARAMETER_TYPE_ANNOTATION);
    }

    return new ParameterImpl(starOrStarStar, name, typeAnnotation, assignToken, defaultValue);
  }

  private Expression numericLiteral(AstNode astNode) {
    return new NumericLiteralImpl(toPyToken(astNode.getToken()));
  }

  private Expression stringLiterals(AstNode astNode) {
    List<StringElement> elements = astNode.getChildren(PythonTokenType.STRING, PythonGrammar.FSTRING).stream()
      .map(this::stringLiteral)
      .filter(Objects::nonNull)
      .collect(Collectors.toList());
    return new StringLiteralImpl(elements);
  }

  private StringElementImpl stringLiteral(AstNode elementNode) {
    Token token = toPyToken(elementNode.getToken());
    if (token == null) {
      return null;
    }
    if (elementNode.is(PythonGrammar.FSTRING)) {
      Token fstringEnd = toPyToken(elementNode.getFirstChild(PythonTokenType.FSTRING_END).getToken());
      List<Tree> fStringMiddles = getFStringMiddles(elementNode);
      return new StringElementImpl(token, fStringMiddles, fstringEnd);
    }
    return new StringElementImpl(token, List.of(), null);
  }

  private List<Tree> getFStringMiddles(AstNode expressionNode) {
    return expressionNode
      .getChildren(PythonGrammar.FSTRING_REPLACEMENT_FIELD, PythonTokenType.FSTRING_MIDDLE)
      .stream()
      .map(fStringMiddle -> {
        if (fStringMiddle.is(PythonGrammar.FSTRING_REPLACEMENT_FIELD)) {
          return formattedExpression(fStringMiddle);
        }
        return stringLiteral(fStringMiddle);
      })
      .filter(Objects::nonNull)
      .collect(Collectors.toList());
  }

  private FormatSpecifier formatSpecifier(AstNode expressionNode) {
    AstNode formatSpecifierNode = expressionNode.getFirstChild(PythonGrammar.FORMAT_SPECIFIER);
    if (formatSpecifierNode == null) {
      return null;
    }
    List<Tree> fStringMiddles = getFStringMiddles(formatSpecifierNode);

    Token columnToken = toPyToken(formatSpecifierNode.getFirstChild(PythonPunctuator.COLON).getToken());
    if (columnToken == null) {
      return null;
    }
    return new FormatSpecifierImpl(columnToken, fStringMiddles);
  }

  private FormattedExpressionImpl formattedExpression(AstNode expressionNode) {
    Expression exp;
    if (expressionNode.hasDirectChildren(PythonGrammar.YIELD_EXPR)) {
      var yieldExpression = expressionNode.getFirstChild(PythonGrammar.YIELD_EXPR);
      exp = expression(yieldExpression);
    } else {
      var expressionsList = expressionNode.getFirstChild(PythonGrammar.TESTLIST_STAR_EXPR);
      exp = exprListOrTestList(expressionsList);
    }
    AstNode equalNode = expressionNode.getFirstChild(PythonPunctuator.ASSIGN);
    Token equalToken = equalNode == null ? null : toPyToken(equalNode.getToken());
    FormatSpecifier formatSpecifier = formatSpecifier(expressionNode);
    Token lCurlyBrace = toPyToken(expressionNode.getFirstChild(PythonPunctuator.LCURLYBRACE).getToken());
    Token rCurlyBrace = toPyToken(expressionNode.getFirstChild(PythonPunctuator.RCURLYBRACE).getToken());

    return getConversionNode(expressionNode)
      .map(conversionNode -> formattedExpressionWithConversion(exp, lCurlyBrace, rCurlyBrace, equalToken, formatSpecifier, conversionNode))
      .orElseGet(() -> new FormattedExpressionImpl(exp, lCurlyBrace, rCurlyBrace, equalToken, formatSpecifier, null, null));
  }

  private FormattedExpressionImpl formattedExpressionWithConversion(Expression exp, Token lCurlyBrace, Token rCurlyBrace, Token equalToken, FormatSpecifier formatSpecifier,
    AstNode conversionNode) {
    Optional<Token> maybeConversionNameToken = getConversionNameToken(conversionNode);
    Token conversionToken = toPyToken(conversionNode.getToken());
    return maybeConversionNameToken
      .map(conversionNameToken -> new FormattedExpressionImpl(exp, lCurlyBrace, rCurlyBrace, equalToken, formatSpecifier, conversionToken, conversionNameToken))
      .orElseGet(() -> new FormattedExpressionImpl(exp, lCurlyBrace, rCurlyBrace, equalToken, formatSpecifier, null, null));
  }

  private static Optional<AstNode> getConversionNode(AstNode expressionNode) {
    return Optional.ofNullable(expressionNode.getFirstChild(GenericTokenType.UNKNOWN_CHAR))
      .filter(node -> "!".equals(node.getTokenValue()));
  }

  private Optional<Token> getConversionNameToken(AstNode conversionNode) {
    return Optional.of(conversionNode)
      .map(AstNode::getNextSibling)
      .filter(node -> node.is(GenericTokenType.IDENTIFIER) && List.of("r", "s", "a").contains(node.getTokenValue()))
      .map(n -> toPyToken(n.getToken()));
  }

  private Token suiteIndent(AstNode suite) {
    return suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.INDENT).getToken());
  }

  private Token suiteNewLine(AstNode suite) {
    return suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.NEWLINE).getToken());
  }

  private Token suiteDedent(AstNode suite) {
    return suite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.DEDENT).getToken());
  }

}
