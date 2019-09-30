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
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.RecognitionException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.DocstringExtractor;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.api.tree.AliasedName;
import org.sonar.python.api.tree.AnnotatedAssignment;
import org.sonar.python.api.tree.AnyParameter;
import org.sonar.python.api.tree.ArgList;
import org.sonar.python.api.tree.Argument;
import org.sonar.python.api.tree.AssertStatement;
import org.sonar.python.api.tree.AssignmentStatement;
import org.sonar.python.api.tree.BreakStatement;
import org.sonar.python.api.tree.CallExpression;
import org.sonar.python.api.tree.ClassDef;
import org.sonar.python.api.tree.CompoundAssignmentStatement;
import org.sonar.python.api.tree.ComprehensionClause;
import org.sonar.python.api.tree.ComprehensionExpression;
import org.sonar.python.api.tree.ComprehensionFor;
import org.sonar.python.api.tree.ConditionalExpression;
import org.sonar.python.api.tree.ContinueStatement;
import org.sonar.python.api.tree.Decorator;
import org.sonar.python.api.tree.DelStatement;
import org.sonar.python.api.tree.DottedName;
import org.sonar.python.api.tree.ElseStatement;
import org.sonar.python.api.tree.ExceptClause;
import org.sonar.python.api.tree.ExecStatement;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.ExpressionList;
import org.sonar.python.api.tree.ExpressionStatement;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.FinallyClause;
import org.sonar.python.api.tree.ForStatement;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.GlobalStatement;
import org.sonar.python.api.tree.IfStatement;
import org.sonar.python.api.tree.ImportFrom;
import org.sonar.python.api.tree.ImportName;
import org.sonar.python.api.tree.ImportStatement;
import org.sonar.python.api.tree.KeyValuePair;
import org.sonar.python.api.tree.LambdaExpression;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.NonlocalStatement;
import org.sonar.python.api.tree.ParameterList;
import org.sonar.python.api.tree.PassStatement;
import org.sonar.python.api.tree.PrintStatement;
import org.sonar.python.api.tree.QualifiedExpression;
import org.sonar.python.api.tree.RaiseStatement;
import org.sonar.python.api.tree.ReturnStatement;
import org.sonar.python.api.tree.SliceItem;
import org.sonar.python.api.tree.SliceList;
import org.sonar.python.api.tree.Statement;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.StringElement;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TryStatement;
import org.sonar.python.api.tree.TypeAnnotation;
import org.sonar.python.api.tree.WithItem;
import org.sonar.python.api.tree.WithStatement;
import org.sonar.python.api.tree.YieldExpression;
import org.sonar.python.api.tree.YieldStatement;

public class PythonTreeMaker {

  public FileInput fileInput(AstNode astNode) {
    List<Statement> statements = getStatements(astNode).stream().map(this::statement).collect(Collectors.toList());
    StatementListImpl statementList = statements.isEmpty() ? null : new StatementListImpl(astNode, statements, toPyToken(astNode.getTokens()));
    Token endOfFile = toPyToken(astNode.getFirstChild(GenericTokenType.EOF).getToken());
    FileInputImpl pyFileInputTree = new FileInputImpl(astNode, statementList, endOfFile, toPyToken(DocstringExtractor.extractDocstring(astNode)));
    setParents(pyFileInputTree);
    return pyFileInputTree;
  }

  private static Token toPyToken(@Nullable com.sonar.sslr.api.Token token) {
    if (token == null) {
      return null;
    }
    return new TokenImpl(token);
  }

  private static List<Token> toPyToken(List<com.sonar.sslr.api.Token> tokens) {
    return tokens.stream().map(TokenImpl::new).collect(Collectors.toList());
  }

  public void setParents(Tree root) {
    for (Tree child : root.children()) {
      if (child != null) {
        ((PyTree) child).setParent(root);
        setParents(child);
      }
    }
  }

  private Statement statement(StatementWithSeparator statementWithSeparator) {
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
    throw new IllegalStateException("Statement " + astNode.getType() + " not correctly translated to strongly typed AST");
  }

  public AnnotatedAssignment annotatedAssignment(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    AstNode annAssign = astNode.getFirstChild(PythonGrammar.ANNASSIGN);
    AstNode colonTokenNode = annAssign.getFirstChild(PythonPunctuator.COLON);
    Expression variable = exprListOrTestList(astNode.getFirstChild(PythonGrammar.TESTLIST_STAR_EXPR));
    Expression annotation = expression(annAssign.getFirstChild(PythonGrammar.TEST));
    AstNode equalTokenNode = annAssign.getFirstChild(PythonPunctuator.ASSIGN);
    Token equalToken = null;
    Expression assignedValue = null;
    if (equalTokenNode != null) {
      equalToken = toPyToken(equalTokenNode.getToken());
      assignedValue = expression(equalTokenNode.getNextSibling());
    }
    return new AnnotatedAssignmentImpl(variable, toPyToken(colonTokenNode.getToken()), annotation, equalToken, assignedValue, separator);
  }

  private StatementList getStatementListFromSuite(AstNode suite) {
    return new StatementListImpl(suite, getStatementsFromSuite(suite), toPyToken(suite.getTokens()));
  }

  private List<Statement> getStatementsFromSuite(AstNode astNode) {
    if (astNode.is(PythonGrammar.SUITE)) {
      List<StatementWithSeparator> statements = getStatements(astNode);
      if (statements.isEmpty()) {
        List<StatementWithSeparator> statementsWithSeparators = getStatementsWithSeparators(astNode);
        return statementsWithSeparators.stream().map(this::statement).collect(Collectors.toList());
      }
      return statements.stream().map(this::statement)
        .collect(Collectors.toList());
    }
    return Collections.emptyList();
  }

  private static List<StatementWithSeparator> getStatements(AstNode astNode) {
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

  private static List<StatementWithSeparator> getStatementsWithSeparators(AstNode stmt) {
    List<StatementWithSeparator> statementsWithSeparators = new ArrayList<>();
    AstNode stmtListNode = stmt.getFirstChild(PythonGrammar.STMT_LIST);
    AstNode newLine = stmt.getFirstChild(PythonTokenType.NEWLINE);
    List<AstNode> children = stmtListNode.getChildren();
    for (int i = 0; i < children.size(); i += 2) {
      AstNode current = children.get(i);
      AstNode separator = newLine;
      AstNode nextSibling = current.getNextSibling();
      if (nextSibling != null) {
        boolean isSecondToLast = (i == children.size() - 2);
        // if current is secondToLast, skip the semicolon and use new line token as separator
        if (!isSecondToLast) {
          separator = nextSibling;
        }
      }
      statementsWithSeparators.add(new StatementWithSeparator(current.getFirstChild(), separator));
    }
    return statementsWithSeparators;
  }

  // Simple statements
  public PrintStatement printStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    List<Expression> expressions = expressionsFromTest(astNode);
    Token separator = statementWithSeparator.separator();
    return new PrintStatementImpl(astNode, toPyToken(astNode.getTokens()).get(0), expressions, separator);
  }

  public ExecStatement execStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Expression expression = expression(astNode.getFirstChild(PythonGrammar.EXPR));
    List<Expression> expressions = expressionsFromTest(astNode);
    Token separator = statementWithSeparator.separator();
    if (expressions.isEmpty()) {
      return new ExecStatementImpl(astNode, toPyToken(astNode.getTokens()).get(0), expression, separator);
    }
    return new ExecStatementImpl(astNode, toPyToken(astNode.getTokens().get(0)), expression,
      expressions.get(0), expressions.size() == 2 ? expressions.get(1) : null, separator);
  }

  public AssertStatement assertStatement(StatementWithSeparator statementWithSeparator) {
    AstNode stmt = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    List<Expression> expressions = expressionsFromTest(stmt);
    Expression condition = expressions.get(0);
    Expression message = null;
    if (expressions.size() > 1) {
      message = expressions.get(1);
    }
    return new AssertStatementImpl(stmt, toPyToken(stmt.getTokens()).get(0), condition, message, separator);
  }

  public PassStatement passStatement(StatementWithSeparator statementWithSeparator) {
    AstNode stmt = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    return new PassStatementImpl(stmt, toPyToken(stmt.getTokens()).get(0), separator);
  }

  public DelStatement delStatement(StatementWithSeparator statementWithSeparator) {
    AstNode stmt = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    List<Expression> expressionTrees = expressionsFromExprList(stmt.getFirstChild(PythonGrammar.EXPRLIST));
    return new DelStatementImpl(stmt, toPyToken(stmt.getTokens()).get(0), expressionTrees, separator);
  }

  public ReturnStatement returnStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    AstNode testListNode = astNode.getFirstChild(PythonGrammar.TESTLIST);
    List<Expression> expressionTrees = Collections.emptyList();
    if (testListNode != null) {
      expressionTrees = expressionsFromTest(testListNode);
    }
    return new ReturnStatementImpl(astNode, toPyToken(astNode.getTokens()).get(0), expressionTrees, separator);
  }

  public YieldStatement yieldStatement(StatementWithSeparator statementWithSeparator) {
    AstNode stmt = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    return new YieldStatementImpl(stmt, yieldExpression(stmt.getFirstChild(PythonGrammar.YIELD_EXPR)), separator);
  }

  public YieldExpression yieldExpression(AstNode astNode) {
    Token yieldKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.YIELD).getToken());
    AstNode nodeContainingExpression = astNode;
    AstNode fromKeyword = astNode.getFirstChild(PythonKeyword.FROM);
    if (fromKeyword == null) {
      nodeContainingExpression = astNode.getFirstChild(PythonGrammar.TESTLIST);
    }
    List<Expression> expressionTrees = Collections.emptyList();
    if (nodeContainingExpression != null) {
      expressionTrees = expressionsFromTest(nodeContainingExpression);
    }
    return new YieldExpressionImpl(astNode, yieldKeyword, fromKeyword == null ? null : toPyToken(fromKeyword.getToken()), expressionTrees);
  }

  public RaiseStatement raiseStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
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
      .collect(Collectors.toList());
    return new RaiseStatementImpl(astNode, toPyToken(astNode.getFirstChild(PythonKeyword.RAISE).getToken()),
      expressionTrees, fromKeyword == null ? null : toPyToken(fromKeyword.getToken()), fromExpression == null ? null : expression(fromExpression), separator);
  }

  public BreakStatement breakStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    return new BreakStatementImpl(astNode, toPyToken(astNode.getToken()), separator);
  }

  public ContinueStatement continueStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    return new ContinueStatementImpl(astNode, toPyToken(astNode.getToken()), separator);
  }

  public ImportStatement importStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    AstNode importStmt = astNode.getFirstChild();
    if (importStmt.is(PythonGrammar.IMPORT_NAME)) {
      return importName(importStmt, separator);
    }
    return importFromStatement(importStmt, separator);
  }

  private ImportName importName(AstNode astNode, Token separator) {
    Token importKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.IMPORT).getToken());
    List<AliasedName> aliasedNames = astNode
      .getFirstChild(PythonGrammar.DOTTED_AS_NAMES)
      .getChildren(PythonGrammar.DOTTED_AS_NAME).stream()
      .map(this::aliasedName)
      .collect(Collectors.toList());
    return new ImportNameImpl(astNode, importKeyword, aliasedNames, separator);
  }

  public ImportFrom importFromStatement(AstNode astNode, Token separator) {
    Token importKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.IMPORT).getToken());
    Token fromKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.FROM).getToken());
    List<Token> dottedPrefixForModule = toPyToken(astNode.getChildren(PythonPunctuator.DOT).stream()
      .map(AstNode::getToken)
      .collect(Collectors.toList()));
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
        .collect(Collectors.toList());
      isWildcardImport = false;
    }
    return new ImportFromImpl(astNode, fromKeyword, dottedPrefixForModule, moduleName, importKeyword, aliasedImportNames, isWildcardImport, separator);
  }

  private AliasedName aliasedName(AstNode astNode) {
    AstNode asKeyword = astNode.getFirstChild(PythonKeyword.AS);
    DottedName dottedName;
    if (astNode.is(PythonGrammar.DOTTED_AS_NAME)) {
      dottedName = dottedName(astNode.getFirstChild(PythonGrammar.DOTTED_NAME));
    } else {
      // astNode is IMPORT_AS_NAME
      AstNode importedName = astNode.getFirstChild(PythonGrammar.NAME);
      dottedName = new DottedNameImpl(astNode, Collections.singletonList(name(importedName)));
    }
    if (asKeyword == null) {
      return new AliasedNameImpl(astNode, null, dottedName, null);
    }
    return new AliasedNameImpl(astNode, toPyToken(asKeyword.getToken()), dottedName, name(astNode.getLastChild(PythonGrammar.NAME)));
  }

  private static DottedName dottedName(AstNode astNode) {
    List<Name> names = astNode
      .getChildren(PythonGrammar.NAME).stream()
      .map(PythonTreeMaker::name)
      .collect(Collectors.toList());
    return new DottedNameImpl(astNode, names);
  }

  public GlobalStatement globalStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    Token globalKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.GLOBAL).getToken());
    List<Name> variables = astNode.getChildren(PythonGrammar.NAME).stream()
      .map(PythonTreeMaker::name)
      .collect(Collectors.toList());
    return new GlobalStatementImpl(astNode, globalKeyword, variables, separator);
  }

  public NonlocalStatement nonlocalStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    Token nonlocalKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.NONLOCAL).getToken());
    List<Name> variables = astNode.getChildren(PythonGrammar.NAME).stream()
      .map(PythonTreeMaker::name)
      .collect(Collectors.toList());
    return new NonlocalStatementImpl(astNode, nonlocalKeyword, variables, separator);
  }
  // Compound statements

  public IfStatement ifStatement(AstNode astNode) {
    Token ifToken = toPyToken(astNode.getTokens().get(0));
    AstNode condition = astNode.getFirstChild(PythonGrammar.TEST);
    Token colon = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    Token indent = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token newLine = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token dedent = suite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.DEDENT).getToken());
    StatementList statements = getStatementListFromSuite(suite);
    AstNode elseSuite = astNode.getLastChild(PythonGrammar.SUITE);
    ElseStatement elseStatement = null;
    if (elseSuite.getPreviousSibling().getPreviousSibling().is(PythonKeyword.ELSE)) {
      elseStatement = elseStatement(elseSuite);
    }
    List<IfStatement> elifBranches = astNode.getChildren(PythonKeyword.ELIF).stream()
      .map(this::elifStatement)
      .collect(Collectors.toList());

    return new IfStatementImpl(ifToken, expression(condition), colon, newLine, indent, statements, dedent, elifBranches, elseStatement);
  }

  private IfStatement elifStatement(AstNode astNode) {
    Token elifToken = toPyToken(astNode.getToken());
    AstNode condition = astNode.getNextSibling();
    AstNode colon = condition.getNextSibling();
    AstNode suite = colon.getNextSibling();
    Token colonToken = toPyToken(colon.getToken());
    Token newLineToken = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token indentToken = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token dedentToken = suite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.DEDENT).getToken());
    StatementList statements = getStatementListFromSuite(suite);
    return new IfStatementImpl(elifToken, expression(condition), colonToken, newLineToken, indentToken, statements, dedentToken);
  }

  private ElseStatement elseStatement(AstNode astNode) {
    Token elseToken = toPyToken(astNode.getPreviousSibling().getPreviousSibling().getToken());
    Token colon = toPyToken(astNode.getPreviousSibling().getToken());
    Token indent = astNode.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(astNode.getFirstChild(PythonTokenType.INDENT).getToken());
    Token newLine = astNode.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(astNode.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token dedent = astNode.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(astNode.getFirstChild(PythonTokenType.DEDENT).getToken());
    StatementList statements = getStatementListFromSuite(astNode);
    return new ElseStatementImpl(elseToken, colon, newLine, indent, statements, dedent);
  }

  public FunctionDef funcDefStatement(AstNode astNode) {
    AstNode decoratorsNode = astNode.getFirstChild(PythonGrammar.DECORATORS);
    List<Decorator> decorators = Collections.emptyList();
    if (decoratorsNode != null) {
      decorators = decoratorsNode.getChildren(PythonGrammar.DECORATOR).stream()
        .map(this::decorator)
        .collect(Collectors.toList());
    }
    Name name = name(astNode.getFirstChild(PythonGrammar.FUNCNAME).getFirstChild(PythonGrammar.NAME));
    ParameterList parameterList = null;
    AstNode typedArgListNode = astNode.getFirstChild(PythonGrammar.TYPEDARGSLIST);
    if (typedArgListNode != null) {
      List<AnyParameter> arguments = typedArgListNode.getChildren(PythonGrammar.TFPDEF).stream()
        .map(this::parameter).collect(Collectors.toList());
      parameterList = new ParameterListImpl(typedArgListNode, arguments);
    }

    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    Token newLine = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token indent = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token dedent = suite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.DEDENT).getToken());
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
    return new FunctionDefImpl(astNode, decorators, asyncToken, toPyToken(defNode.getToken()), name, lPar, parameterList, rPar,
      returnType, colon, newLine, indent, body, dedent, isMethodDefinition(astNode), toPyToken(DocstringExtractor.extractDocstring(astNode)));
  }

  private Decorator decorator(AstNode astNode) {
    Token atToken = toPyToken(astNode.getFirstChild(PythonPunctuator.AT).getToken());
    DottedName dottedName = dottedName(astNode.getFirstChild(PythonGrammar.DOTTED_NAME));
    Token lPar = astNode.getFirstChild(PythonPunctuator.LPARENTHESIS) == null ? null : toPyToken(astNode.getFirstChild(PythonPunctuator.LPARENTHESIS).getToken());
    Token rPar = astNode.getFirstChild(PythonPunctuator.RPARENTHESIS) == null ? null : toPyToken(astNode.getFirstChild(PythonPunctuator.RPARENTHESIS).getToken());
    ArgList argListTree = argList(astNode.getFirstChild(PythonGrammar.ARGLIST));
    Token newLineToken = astNode.getFirstChild(PythonTokenType.NEWLINE) == null ? null : toPyToken(astNode.getFirstChild(PythonTokenType.NEWLINE).getToken());
    return new DecoratorImpl(astNode, atToken, dottedName, lPar, argListTree, rPar, newLineToken);
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
        .collect(Collectors.toList());
    }
    Name name = name(astNode.getFirstChild(PythonGrammar.CLASSNAME).getFirstChild(PythonGrammar.NAME));
    ArgList args = null;
    AstNode leftPar = astNode.getFirstChild(PythonPunctuator.LPARENTHESIS);
    if (leftPar != null) {
      args = argList(astNode.getFirstChild(PythonGrammar.ARGLIST));
    }
    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    StatementList body = getStatementListFromSuite(suite);
    Token indent = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token newLine = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token dedent = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.DEDENT).getToken());
    Token classToken = toPyToken(astNode.getFirstChild(PythonKeyword.CLASS).getToken());
    AstNode rightPar = astNode.getFirstChild(PythonPunctuator.RPARENTHESIS);
    Token colon = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    return new ClassDefImpl(astNode, decorators, classToken, name,
      leftPar != null ? toPyToken(leftPar.getToken()) : null, args, rightPar != null ? toPyToken(rightPar.getToken()) : null,
      colon, newLine, indent, body, dedent, toPyToken(DocstringExtractor.extractDocstring(astNode)));
  }

  private static Name name(AstNode astNode) {
    return new NameImpl(toPyToken(astNode.getFirstChild(GenericTokenType.IDENTIFIER).getToken()), astNode.getParent().is(PythonGrammar.ATOM));
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
    List<Expression> expressions = expressionsFromExprList(forStatementNode.getFirstChild(PythonGrammar.EXPRLIST));
    List<Expression> testExpressions = expressionsFromTest(forStatementNode.getFirstChild(PythonGrammar.TESTLIST));
    AstNode firstSuite = forStatementNode.getFirstChild(PythonGrammar.SUITE);
    Token firstIndent = firstSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(firstSuite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token firstNewLine = firstSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(firstSuite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token firstDedent = firstSuite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(firstSuite.getFirstChild(PythonTokenType.DEDENT).getToken());
    StatementList body = getStatementListFromSuite(firstSuite);
    AstNode lastSuite = forStatementNode.getLastChild(PythonGrammar.SUITE);
    Token lastIndent = lastSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(lastSuite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token lastNewLine = lastSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(lastSuite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token lastDedent = lastSuite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(lastSuite.getFirstChild(PythonTokenType.DEDENT).getToken());
    AstNode elseKeywordNode = forStatementNode.getFirstChild(PythonKeyword.ELSE);
    Token elseKeyword = null;
    Token elseColonKeyword = null;
    if (elseKeywordNode != null) {
      elseKeyword = toPyToken(elseKeywordNode.getToken());
      elseColonKeyword = toPyToken(elseKeywordNode.getNextSibling().getToken());
    }
    StatementList elseBody = lastSuite == firstSuite ? null : getStatementListFromSuite(lastSuite);
    return new ForStatementImpl(forStatementNode, forKeyword, expressions, inKeyword, testExpressions, colon, firstNewLine, firstIndent,
      body, firstDedent, elseKeyword, elseColonKeyword, lastNewLine, lastIndent, elseBody, lastDedent, asyncToken);
  }

  public WhileStatementImpl whileStatement(AstNode astNode) {
    Token whileKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.WHILE).getToken());
    Token colon = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    Expression condition = expression(astNode.getFirstChild(PythonGrammar.TEST));
    AstNode firstSuite = astNode.getFirstChild(PythonGrammar.SUITE);
    Token firstIndent = firstSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(firstSuite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token firstNewLine = firstSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(firstSuite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token firstDedent = firstSuite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(firstSuite.getFirstChild(PythonTokenType.DEDENT).getToken());
    StatementList body = getStatementListFromSuite(firstSuite);
    AstNode lastSuite = astNode.getLastChild(PythonGrammar.SUITE);
    Token lastIndent = lastSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(lastSuite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token lastNewLine = lastSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(lastSuite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token lastDedent = lastSuite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(lastSuite.getFirstChild(PythonTokenType.DEDENT).getToken());
    AstNode elseKeywordNode = astNode.getFirstChild(PythonKeyword.ELSE);
    Token elseKeyword = null;
    Token elseColonKeyword = null;
    if (elseKeywordNode != null) {
      elseKeyword = toPyToken(elseKeywordNode.getToken());
      elseColonKeyword = toPyToken(elseKeywordNode.getNextSibling().getToken());
    }
    StatementList elseBody = lastSuite == firstSuite ? null : getStatementListFromSuite(lastSuite);
    return new WhileStatementImpl(astNode, whileKeyword, condition, colon, firstNewLine, firstIndent,
      body, firstDedent, elseKeyword, elseColonKeyword, lastNewLine, lastIndent, elseBody, lastDedent);
  }


  public ExpressionStatement expressionStatement(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    List<Expression> expressions = astNode.getFirstChild(PythonGrammar.TESTLIST_STAR_EXPR).getChildren(PythonGrammar.TEST, PythonGrammar.STAR_EXPR).stream()
      .map(this::expression)
      .collect(Collectors.toList());
    return new ExpressionStatementImpl(astNode, expressions, separator);
  }

  public AssignmentStatement assignment(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    List<Token> assignTokens = new ArrayList<>();
    List<ExpressionList> lhsExpressions = new ArrayList<>();
    List<AstNode> assignNodes = astNode.getChildren(PythonPunctuator.ASSIGN);
    for (AstNode assignNode : assignNodes) {
      assignTokens.add(toPyToken(assignNode.getToken()));
      lhsExpressions.add(expressionList(assignNode.getPreviousSibling()));
    }
    AstNode assignedValueNode = assignNodes.get(assignNodes.size() - 1).getNextSibling();
    Expression assignedValue = assignedValueNode.is(PythonGrammar.YIELD_EXPR) ? yieldExpression(assignedValueNode) : exprListOrTestList(assignedValueNode);
    return new AssignmentStatementImpl(astNode, assignTokens, lhsExpressions, assignedValue, separator);
  }

  public CompoundAssignmentStatement compoundAssignment(StatementWithSeparator statementWithSeparator) {
    AstNode astNode = statementWithSeparator.statement();
    Token separator = statementWithSeparator.separator();
    AstNode augAssignNodes = astNode.getFirstChild(PythonGrammar.AUGASSIGN);
    Expression lhsExpression = exprListOrTestList(augAssignNodes.getPreviousSibling());
    AstNode rhsAstNode = augAssignNodes.getNextSibling();
    Expression rhsExpression;
    if (rhsAstNode.is(PythonGrammar.YIELD_EXPR)) {
      rhsExpression = yieldExpression(rhsAstNode);
    } else {
      rhsExpression = exprListOrTestList(rhsAstNode);
    }
    return new CompoundAssignmentStatementImpl(astNode, lhsExpression, toPyToken(augAssignNodes.getToken()), rhsExpression, separator);
  }

  private ExpressionList expressionList(AstNode astNode) {
    if (astNode.is(PythonGrammar.TESTLIST_STAR_EXPR, PythonGrammar.TESTLIST_COMP)) {
      List<Expression> expressions = astNode.getChildren(PythonGrammar.TEST, PythonGrammar.STAR_EXPR).stream()
        .map(this::expression)
        .collect(Collectors.toList());
      return new ExpressionListImpl(astNode, expressions);
    }
    return new ExpressionListImpl(astNode, Collections.singletonList(expression(astNode)));
  }

  public TryStatement tryStatement(AstNode astNode) {
    Token tryKeyword = toPyToken(astNode.getFirstChild(PythonKeyword.TRY).getToken());
    Token colon = toPyToken(astNode.getFirstChild(PythonPunctuator.COLON).getToken());
    AstNode firstSuite = astNode.getFirstChild(PythonGrammar.SUITE);
    Token indent = firstSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(firstSuite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token newLine = firstSuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(firstSuite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token dedent = firstSuite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(firstSuite.getFirstChild(PythonTokenType.DEDENT).getToken());
    StatementList tryBody = getStatementListFromSuite(firstSuite);
    List<ExceptClause> exceptClauseTrees = astNode.getChildren(PythonGrammar.EXCEPT_CLAUSE).stream()
      .map(except -> {
        AstNode suite = except.getNextSibling().getNextSibling();
        return exceptClause(except, getStatementListFromSuite(suite));
      })
      .collect(Collectors.toList());
    FinallyClause finallyClause = null;
    AstNode finallyNode = astNode.getFirstChild(PythonKeyword.FINALLY);
    if (finallyNode != null) {
      Token finallyColon = toPyToken(finallyNode.getNextSibling().getToken());
      AstNode finallySuite = finallyNode.getNextSibling().getNextSibling();
      StatementList body = getStatementListFromSuite(finallySuite);
      Token finallyIndent = finallySuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(finallySuite.getFirstChild(PythonTokenType.INDENT).getToken());
      Token finallyNewLine = finallySuite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(finallySuite.getFirstChild(PythonTokenType.NEWLINE).getToken());
      Token finallyDedent = finallySuite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(finallySuite.getFirstChild(PythonTokenType.DEDENT).getToken());
      finallyClause = new FinallyClauseImpl(toPyToken(finallyNode.getToken()), finallyColon, finallyNewLine, finallyIndent, body, finallyDedent);
    }
    ElseStatement elseStatementTree = null;
    AstNode elseNode = astNode.getFirstChild(PythonKeyword.ELSE);
    if (elseNode != null) {
      elseStatementTree = elseStatement(elseNode.getNextSibling().getNextSibling());
    }
    return new TryStatementImpl(astNode, tryKeyword, colon, newLine, indent, tryBody, dedent, exceptClauseTrees, finallyClause, elseStatementTree);
  }

  public WithStatement withStatement(AstNode astNode) {
    AstNode withStmtNode = astNode;
    Token asyncKeyword = null;
    if (astNode.is(PythonGrammar.ASYNC_STMT)) {
      withStmtNode = astNode.getFirstChild(PythonGrammar.WITH_STMT);
      asyncKeyword = toPyToken(astNode.getFirstChild().getToken());
    }
    List<WithItem> withItems = withItems(withStmtNode.getChildren(PythonGrammar.WITH_ITEM));
    AstNode suite = withStmtNode.getFirstChild(PythonGrammar.SUITE);
    Token withKeyword = toPyToken(withStmtNode.getToken());
    Token indent = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token newLine = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token dedent = suite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.DEDENT).getToken());
    Token colon = toPyToken(suite.getPreviousSibling().getToken());
    StatementList statements = getStatementListFromSuite(suite);
    return new WithStatementImpl(withStmtNode, withKeyword, withItems, colon, newLine, indent, statements, dedent, asyncKeyword);
  }

  private List<WithItem> withItems(List<AstNode> withItems) {
    return withItems.stream().map(this::withItem).collect(Collectors.toList());
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
    return new WithStatementImpl.WithItemImpl(withItem, test, as, expr);
  }

  private ExceptClause exceptClause(AstNode except, StatementList body) {
    Token colon = toPyToken(except.getNextSibling().getToken());
    AstNode suite = except.getNextSibling().getNextSibling();
    Token exceptKeyword = toPyToken(except.getFirstChild(PythonKeyword.EXCEPT).getToken());
    Token indent = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.INDENT).getToken());
    Token newLine = suite.getFirstChild(PythonTokenType.INDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.NEWLINE).getToken());
    Token dedent = suite.getFirstChild(PythonTokenType.DEDENT) == null ? null : toPyToken(suite.getFirstChild(PythonTokenType.DEDENT).getToken());
    AstNode exceptionNode = except.getFirstChild(PythonGrammar.TEST);
    if (exceptionNode == null) {
      return new ExceptClauseImpl(exceptKeyword, colon, newLine, indent, body, dedent);
    }
    AstNode asNode = except.getFirstChild(PythonKeyword.AS);
    AstNode commaNode = except.getFirstChild(PythonPunctuator.COMMA);
    if (asNode != null || commaNode != null) {
      Expression exceptionInstance = expression(except.getLastChild(PythonGrammar.TEST));
      Token asNodeToken = asNode != null ? toPyToken(asNode.getToken()) : null;
      Token commaNodeToken = commaNode != null ? toPyToken(commaNode.getToken()) : null;
      return new ExceptClauseImpl(exceptKeyword, colon, newLine, indent, body, dedent, expression(exceptionNode), asNodeToken, commaNodeToken, exceptionInstance);
    }
    return new ExceptClauseImpl(exceptKeyword, colon, newLine, indent, body, dedent, expression(exceptionNode));
  }

  // expressions

  private List<Expression> expressionsFromTest(AstNode astNode) {
    return astNode.getChildren(PythonGrammar.TEST).stream().map(this::expression).collect(Collectors.toList());
  }

  private List<Expression> expressionsFromExprList(AstNode firstChild) {
    return firstChild
      .getChildren(PythonGrammar.EXPR, PythonGrammar.STAR_EXPR)
      .stream().map(this::expression).collect(Collectors.toList());
  }

  private Expression exprListOrTestList(AstNode exprListOrTestList) {
    List<Expression> expressions = exprListOrTestList
      .getChildren(PythonGrammar.EXPR, PythonGrammar.STAR_EXPR, PythonGrammar.TEST).stream()
      .map(this::expression)
      .collect(Collectors.toList());
    List<AstNode> commas = exprListOrTestList.getChildren(PythonPunctuator.COMMA);
    if (commas.isEmpty()) {
      return expressions.get(0);
    }
    List<Token> commaTokens = toPyToken(commas.stream().map(AstNode::getToken).collect(Collectors.toList()));
    return new TupleImpl(exprListOrTestList, null, expressions, commaTokens, null);
  }

  Expression expression(AstNode astNode) {
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
    if (astNode.is(PythonGrammar.ATOM) && astNode.getFirstChild().is(PythonTokenType.STRING)) {
      return stringLiteral(astNode);
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
    if (astNode.is(PythonGrammar.ATTRIBUTE_REF)) {
      return qualifiedExpression(astNode);
    }
    if (astNode.is(PythonGrammar.CALL_EXPR)) {
      return callExpression(astNode);
    }
    if (astNode.is(PythonGrammar.EXPR, PythonGrammar.TEST, PythonGrammar.TEST_NOCOND)) {
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
      return new UnaryExpressionImpl(astNode, toPyToken(astNode.getFirstChild().getToken()), expression(astNode.getLastChild()));
    }
    if (astNode.is(PythonGrammar.STAR_EXPR)) {
      return new StarredExpressionImpl(astNode, toPyToken(astNode.getToken()), expression(astNode.getLastChild()));
    }
    if (astNode.is(PythonGrammar.SUBSCRIPTION_OR_SLICING)) {
      Expression baseExpr = expression(astNode.getFirstChild(PythonGrammar.ATOM));
      Token leftBracket = toPyToken(astNode.getFirstChild(PythonPunctuator.LBRACKET).getToken());
      Token rightBracket = toPyToken(astNode.getFirstChild(PythonPunctuator.RBRACKET).getToken());
      return subscriptionOrSlicing(baseExpr, leftBracket, astNode, rightBracket);
    }
    if (astNode.is(PythonKeyword.NONE)) {
      return new NoneExpressionImpl(astNode, toPyToken(astNode.getToken()));
    }
    if (astNode.is(PythonGrammar.ELLIPSIS)) {
      return new EllipsisExpressionImpl(astNode);
    }
    throw new IllegalStateException("Expression " + astNode.getType() + " not correctly translated to strongly typed AST");
  }

  private Expression repr(AstNode astNode) {
    Token openingBacktick = toPyToken(astNode.getFirstChild(PythonPunctuator.BACKTICK).getToken());
    Token closingBacktick = toPyToken(astNode.getLastChild(PythonPunctuator.BACKTICK).getToken());
    List<Expression> expressions = astNode.getChildren(PythonGrammar.TEST).stream().map(this::expression).collect(Collectors.toList());
    ExpressionList expressionListTree = new ExpressionListImpl(expressions);
    return new ReprExpressionImpl(astNode, openingBacktick, expressionListTree, closingBacktick);
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
        Expression resultExpression = expression(dictOrSetMaker.getFirstChild(PythonGrammar.TEST, PythonGrammar.STAR_EXPR));
        return new ComprehensionExpressionImpl(Tree.Kind.SET_COMPREHENSION, lCurlyBrace, resultExpression, compFor, rCurlyBrace);
      }
    }
    List<Token> commas = toPyToken(dictOrSetMaker.getChildren(PythonPunctuator.COMMA).stream().map(AstNode::getToken).collect(Collectors.toList()));
    if (dictOrSetMaker.hasDirectChildren(PythonPunctuator.COLON) || dictOrSetMaker.hasDirectChildren(PythonPunctuator.MUL_MUL)) {
      List<KeyValuePair> keyValuePairTrees = new ArrayList<>();
      List<AstNode> children = dictOrSetMaker.getChildren();
      int index = 0;
      while (index < children.size()) {
        AstNode currentChild = children.get(index);
        if (currentChild.is(PythonPunctuator.MUL_MUL)) {
          keyValuePairTrees.add(new KeyValuePairImpl(toPyToken(currentChild.getToken()), expression(children.get(index + 1))));
          index += 3;
        } else {
          keyValuePairTrees.add(new KeyValuePairImpl(expression(currentChild), toPyToken(children.get(index + 1).getToken()), expression(children.get(index + 2))));
          index += 4;
        }
      }
      return new DictionaryLiteralImpl(lCurlyBrace, commas, keyValuePairTrees, rCurlyBrace);
    }
    List<Expression> expressions = dictOrSetMaker.getChildren(PythonGrammar.TEST, PythonGrammar.STAR_EXPR).stream().map(this::expression).collect(Collectors.toList());
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
      return new TupleImpl(atom, lPar, Collections.emptyList(), Collections.emptyList(), rPar);
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

    List<Token> commaTokens = toPyToken(commas.stream().map(AstNode::getToken).collect(Collectors.toList()));
    return new TupleImpl(atom, lPar, expressionList.expressions(), commaTokens, rPar);
  }

  private ConditionalExpression conditionalExpression(AstNode astNode) {
    List<AstNode> children = astNode.getChildren();
    Expression trueExpression = expression(children.get(0));
    Token ifToken = toPyToken(astNode.getFirstChild(PythonKeyword.IF).getToken());
    Expression condition = expression(children.get(2));
    Token elseToken = toPyToken(astNode.getFirstChild(PythonKeyword.ELSE).getToken());
    Expression falseExpression = expression(children.get(4));
    return new ConditionalExpressionImpl(astNode, trueExpression, ifToken, condition, elseToken, falseExpression);
  }

  private Expression powerExpression(AstNode astNode) {
    Expression expr = expression(astNode.getFirstChild(PythonGrammar.CALL_EXPR, PythonGrammar.ATTRIBUTE_REF, PythonGrammar.ATOM));
    for (AstNode trailer : astNode.getChildren(PythonGrammar.TRAILER)) {
      expr = withTrailer(expr, trailer);
    }
    if (astNode.getFirstChild().is(GenericTokenType.IDENTIFIER)) {
      expr = new AwaitExpressionImpl(astNode, toPyToken(astNode.getFirstChild().getToken()), expr);
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
      Token leftPar = toPyToken(firstChild.getToken());
      Token rightPar = toPyToken(trailer.getFirstChild(PythonPunctuator.RPARENTHESIS).getToken());
      return new CallExpressionImpl(expr, argList(argListNode), leftPar, rightPar);

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
      if (colon == null) {
        slices.add(expression(subscript.getFirstChild(PythonGrammar.TEST)));
      } else {
        slices.add(sliceItem(subscript));
      }
    }

    // https://docs.python.org/3/reference/expressions.html#slicings
    // "There is ambiguity in the formal syntax here"
    // "a subscription takes priority over the interpretation as a slicing (this is the case if the slice list contains no proper slice)"
    if (slices.stream().anyMatch(s -> Tree.Kind.SLICE_ITEM.equals(s.getKind()))) {
      List<Token> separators = toPyToken(subscriptList.getChildren(PythonPunctuator.COMMA).stream()
        .map(AstNode::getToken)
        .collect(Collectors.toList()));
      SliceList sliceList = new SliceListImpl(subscriptList, slices, separators);
      return new SliceExpressionImpl(expr, leftBracket, sliceList, rightBracket);

    } else {
      List<Expression> expressions = slices.stream().map(Expression.class::cast).collect(Collectors.toList());
      ExpressionList subscripts = new ExpressionListImpl(expressions);
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
    return new SliceItemImpl(subscript, lowerBound, toPyToken(boundSeparator.getToken()), upperBound, strideSeparator, stride);
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
        Expression resultExpression = expression(testListComp.getFirstChild(PythonGrammar.TEST, PythonGrammar.STAR_EXPR));
        return new ComprehensionExpressionImpl(Tree.Kind.LIST_COMPREHENSION, leftBracket, resultExpression, compFor(compForNode), rightBracket);
      }
      elements = expressionList(testListComp);
    } else {
      elements = new ExpressionListImpl(astNode, Collections.emptyList());
    }
    return new ListLiteralImpl(astNode, leftBracket, elements, rightBracket);
  }

  private ComprehensionFor compFor(AstNode compFor) {
    Expression expression = exprListOrTestList(compFor.getFirstChild(PythonGrammar.EXPRLIST));
    Token forToken = toPyToken(compFor.getFirstChild(PythonKeyword.FOR).getToken());
    Token inToken = toPyToken(compFor.getFirstChild(PythonKeyword.IN).getToken());
    Expression iterable = exprListOrTestList(compFor.getFirstChild(PythonGrammar.TESTLIST));
    ComprehensionClause nested = compClause(compFor.getFirstChild(PythonGrammar.COMP_ITER));
    return new ComprehensionForImpl(compFor, forToken, expression, inToken, iterable, nested);
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
      return new ComprehensionIfImpl(child, ifToken, condition, nestedClause);
    }
  }

  public QualifiedExpression qualifiedExpression(AstNode astNode) {
    Expression qualifier = expression(astNode.getFirstChild());
    List<AstNode> names = astNode.getChildren(PythonGrammar.NAME);
    AstNode lastNameNode = astNode.getLastChild();
    for (AstNode nameNode : names) {
      if (nameNode != lastNameNode) {
        qualifier = new QualifiedExpressionImpl(name(nameNode), qualifier, toPyToken(nameNode.getPreviousSibling().getToken()));
      }
    }
    return new QualifiedExpressionImpl(name(lastNameNode), qualifier, toPyToken(lastNameNode.getPreviousSibling().getToken()));
  }

  public CallExpression callExpression(AstNode astNode) {
    Expression callee = expression(astNode.getFirstChild());
    AstNode argListNode = astNode.getFirstChild(PythonGrammar.ARGLIST);
    ArgList argumentList = argList(argListNode);
    if (argumentList != null) {
      checkGeneratorExpressionInArgument(argumentList.arguments());
    }
    Token leftPar = toPyToken(astNode.getFirstChild(PythonPunctuator.LPARENTHESIS).getToken());
    Token rightPar = toPyToken(astNode.getFirstChild(PythonPunctuator.RPARENTHESIS).getToken());
    return new CallExpressionImpl(astNode, callee, argumentList, leftPar, rightPar);
  }

  @CheckForNull
  private ArgList argList(@Nullable AstNode argList) {
    if (argList != null) {
      List<Argument> arguments = argList.getChildren(PythonGrammar.ARGUMENT).stream()
        .map(this::argument)
        .collect(Collectors.toList());
      return new ArgListImpl(argList, arguments);
    }
    return null;
  }

  /*
   * Post Condition on Generator Expression: parentheses can be omitted on calls with only one argument.
   * https://docs.python.org/3/reference/expressions.html#grammar-token-generator-expression
   */
  private static void checkGeneratorExpressionInArgument(List<Argument> arguments) {
    List<Argument> nonParenthesizedGeneratorExpressions = arguments.stream()
      .filter(arg -> arg.expression().is(Tree.Kind.GENERATOR_EXPR) && !arg.expression().firstToken().value().equals("("))
      .collect(Collectors.toList());
    if (!nonParenthesizedGeneratorExpressions.isEmpty() && arguments.size() > 1) {
      int line = nonParenthesizedGeneratorExpressions.get(0).firstToken().line();
      throw new RecognitionException(line, "Parse error at line " + line + ": Generator expression must be parenthesized if not sole argument.");
    }
  }

  public Argument argument(AstNode astNode) {
    AstNode compFor = astNode.getFirstChild(PythonGrammar.COMP_FOR);
    if (compFor != null) {
      Expression expression = expression(astNode.getFirstChild());
      ComprehensionExpression comprehension =
        new ComprehensionExpressionImpl(Tree.Kind.GENERATOR_EXPR, expression.firstToken(), expression, compFor(compFor), toPyToken(compFor.getLastToken()));
      return new ArgumentImpl(astNode, comprehension, null, null);
    }
    AstNode assign = astNode.getFirstChild(PythonPunctuator.ASSIGN);
    Token star = astNode.getFirstChild(PythonPunctuator.MUL) == null ? null : toPyToken(astNode.getFirstChild(PythonPunctuator.MUL).getToken());
    Token starStar = astNode.getFirstChild(PythonPunctuator.MUL_MUL) == null ? null : toPyToken(astNode.getFirstChild(PythonPunctuator.MUL_MUL).getToken());
    Expression arg = expression(astNode.getLastChild(PythonGrammar.TEST));
    if (assign != null) {
      // Keyword in argument list must be an identifier.
      AstNode nameNode = astNode.getFirstChild(PythonGrammar.TEST).getFirstChild(PythonGrammar.ATOM).getFirstChild(PythonGrammar.NAME);
      return new ArgumentImpl(astNode, name(nameNode), arg, toPyToken(assign.getToken()), star, starStar);
    }
    return new ArgumentImpl(astNode, arg, star, starStar);
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
      List<AnyParameter> parameters = varArgsListNode.getChildren(PythonGrammar.FPDEF, PythonGrammar.NAME).stream()
        .map(this::parameter).collect(Collectors.toList());
      argListTree = new ParameterListImpl(varArgsListNode, parameters);
    }

    return new LambdaExpressionImpl(astNode, lambdaKeyword, colonToken, body, argListTree);
  }

  private AnyParameter parameter(AstNode parameter) {
    AstNode prevSibling = parameter.getPreviousSibling();

    if (parameter.is(PythonGrammar.NAME)) {
      return new ParameterImpl(parameter, toPyToken(prevSibling.getToken()), name(parameter), null, null, null);
    }

    // parameter is FPDEF or TFPDEF

    AstNode paramList = parameter.getFirstChild(PythonGrammar.TFPLIST, PythonGrammar.FPLIST);
    // Python 2 only, PEP 3113: Tuple parameter unpacking removed
    if (paramList != null) {
      List<AnyParameter> params = paramList.getChildren(PythonGrammar.TFPDEF, PythonGrammar.FPDEF).stream()
        .map(this::parameter)
        .collect(Collectors.toList());
      List<Token> commas = toPyToken(paramList.getChildren(PythonPunctuator.COMMA).stream().map(AstNode::getToken).collect(Collectors.toList()));
      return new TupleParameterImpl(parameter, params, commas);
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
    AstNode testNode = parameter.getFirstChild(PythonGrammar.TEST);
    if (testNode != null) {
      Token colonToken = toPyToken(parameter.getFirstChild(PythonPunctuator.COLON).getToken());
      typeAnnotation = new TypeAnnotationImpl(colonToken, expression(testNode));
    }

    return new ParameterImpl(parameter, starOrStarStar, name, typeAnnotation, assignToken, defaultValue);
  }

  private static Expression numericLiteral(AstNode astNode) {
    return new NumericLiteralImpl(toPyToken(astNode.getToken()));
  }

  private static Expression stringLiteral(AstNode astNode) {
    List<StringElement> stringElements = astNode.getChildren(PythonTokenType.STRING).stream().map(StringElementImpl::new).collect(Collectors.toList());
    return new StringLiteralImpl(astNode, stringElements);
  }
}
