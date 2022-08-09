/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.checks.tests;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;

@Rule(key = "S5918")
public class TestShouldBeSkippedExplicitlyCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Skip this test explicitly.";
  private static final Set<String> supportedAssertMethods = new HashSet<>(Arrays.asList("assertEqual", "assertNotEqual", "assertTrue",
    "assertFalse", "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn", "assertNotIn", "assertIsInstance",
    "assertNotIsInstance", "assertRaises", "assertRaisesRegexp", "assertAlmostEqual", "assertNotAlmostEqual", "assertGreater",
    "assertGreaterEqual", "assertLess", "assertLessEqual", "assertRegexpMatches", "assertNotRegexpMatches", "assertItemsEqual",
    "assertDictContainsSubset"));

  private static class ReturnAnalyze {
    final Set<Tree> returnParentNodes = new HashSet<>();

    boolean functionNameStartWithTest = false;
    boolean isInIfStatement = false;
    boolean isInConditionWithCallingStatement = false;
    boolean hasAnAssertStatementBefore = false;
    boolean hasAnyAssertStatement = false;
    boolean hasACallExpressionBefore = false;
    boolean isReturnNodeFirstConditionalStatement = true;

    FunctionDef functionDef = null;
  }

  @Override
  public void initialize(Context context) {
    // Look for invalid return in test methods
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_STMT, ctx -> {
      ReturnStatement returnStatement = (ReturnStatement) ctx.syntaxNode();
      ReturnAnalyze analyzeResult = analyze(returnStatement);

      if (isTheReturnAnIssue(analyzeResult)) {
        ctx.addIssue(returnStatement, MESSAGE);
      }
    });
  }

  private static boolean isTheReturnAnIssue(ReturnAnalyze analyzeResult) {
    if (!analyzeResult.functionNameStartWithTest) {
      return false;
    }
    // Not in an if statement, out of scope (detected by S1763)
    if (!analyzeResult.isInIfStatement) {
      return false;
    }
    if (analyzeResult.hasAnAssertStatementBefore) {
      return false;
    }
    if (!analyzeResult.hasAnyAssertStatement) {
      return false;
    }
    if (analyzeResult.hasACallExpressionBefore) {
      return false;
    }
    if (!analyzeResult.isInConditionWithCallingStatement) {
      return false;
    }
    if (!analyzeResult.isReturnNodeFirstConditionalStatement) {
      return false;
    }
    return true;
  }

  private static ReturnAnalyze analyze(ReturnStatement returnStatement) {
    ReturnAnalyze returnAnalyze = new ReturnAnalyze();

    analyzeFunction(returnStatement, returnAnalyze);
    analyzeStatements(returnAnalyze);

    return returnAnalyze;
  }

  private static void analyzeFunction(ReturnStatement returnStatement, ReturnAnalyze returnAnalyze) {
    // look for the function
    Tree parent = returnStatement;
    boolean hasFoundFunction = false;

    do {
      returnAnalyze.returnParentNodes.add(parent);

      if (parent.is(Tree.Kind.FUNCDEF)) {
        returnAnalyze.functionDef = (FunctionDef) parent;
        returnAnalyze.functionNameStartWithTest = returnAnalyze.functionDef.name().name().startsWith("test");
        hasFoundFunction = true;
      } else if (parent.is(Tree.Kind.IF_STMT)) {
        returnAnalyze.isInIfStatement = true;
        if (hasACallStatementInIfStatement((IfStatement) parent)) {
          returnAnalyze.isInConditionWithCallingStatement = true;
        }
      }

      parent = parent.parent();
    } while(parent != null && !hasFoundFunction);
  }

  private static boolean hasACallStatementInIfStatement(IfStatement ifStatement) {
    return isACallExpression(ifStatement.condition());
  }

  private static boolean isACallExpression(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      return true;
    }

    if (expression.is(Tree.Kind.NOT)) {
      UnaryExpression unaryExpression = (UnaryExpression) expression;
      return isACallExpression(unaryExpression.expression());
    }

    if (expression.is(Tree.Kind.BITWISE_AND, Tree.Kind.BITWISE_OR, Tree.Kind.BITWISE_XOR, Tree.Kind.BITWISE_COMPLEMENT,
      Tree.Kind.AND, Tree.Kind.OR)) {
      BinaryExpression binaryExpression = (BinaryExpression) expression;
      return isACallExpression(binaryExpression.leftOperand()) || isACallExpression(binaryExpression.rightOperand());
    }

    return false;
  }

  private static void analyzeStatements(ReturnAnalyze returnAnalyze) {
    if (returnAnalyze.functionDef != null) {
      analyzeStatements(returnAnalyze.functionDef.body(), returnAnalyze, true);
    }
  }

  private static void analyzeStatements(StatementList statements, ReturnAnalyze result, boolean isBeforeReturnNode) {
    for (Statement statement : statements.statements()) {
      if (result.returnParentNodes.contains(statement)) {
        isBeforeReturnNode = false;
      }

      computeAssertStatementsValues(result, statement, isBeforeReturnNode);
      computeCallExpression(result, statement, isBeforeReturnNode);
      if (statement.is(Tree.Kind.IF_STMT) && isBeforeReturnNode && !result.returnParentNodes.contains(statement)) {
        result.isReturnNodeFirstConditionalStatement = false;
      }

      // looking for inner statement list in IF/ELSIF/ELSE to do recursive calls
      analyseIfStatementList(result, statement, isBeforeReturnNode);
    }
  }

  private static void analyseIfStatementList(ReturnAnalyze result, Statement statement, boolean isBeforeReturnNode) {
    if (statement.is(Tree.Kind.IF_STMT)) {
      IfStatement ifStatement = (IfStatement) statement;
      analyzeStatements(ifStatement.body(), result, isBeforeReturnNode);

      for (IfStatement elifStatement : ifStatement.elifBranches()) {
        analyzeStatements(elifStatement.body(), result, isBeforeReturnNode);
      }

      ElseClause elseClause = ifStatement.elseBranch();
      if (elseClause != null) {
        analyzeStatements(elseClause.body(), result, isBeforeReturnNode);
      }
    }
  }

  private static void computeAssertStatementsValues(ReturnAnalyze result, Statement statement, boolean isBeforeReturnNode) {
    if (isAnAssertStatement(statement)) {
      if (isBeforeReturnNode) {
        result.hasAnAssertStatementBefore = true;
      }
      result.hasAnyAssertStatement = true;
    }
  }

  private static void computeCallExpression(ReturnAnalyze result, Statement statement, boolean isBeforeReturnNode) {
    if (statement.is(Tree.Kind.EXPRESSION_STMT) && isBeforeReturnNode) {
      ExpressionStatement expressionStatement = (ExpressionStatement) statement;
      for (Expression expression : expressionStatement.expressions()) {
        if (expression.is(Tree.Kind.CALL_EXPR)) {
          result.hasACallExpressionBefore = true;
          break;
        }
      }
    }
  }

  private static boolean isAnAssertStatement(Statement statement) {
    // Native assert statement
    if (statement.is(Tree.Kind.ASSERT_STMT)) {
      return true;
    }

    // Pytest self.assertXXX methods
    if (statement.is(Tree.Kind.EXPRESSION_STMT) && ((ExpressionStatement) statement).expressions().size() == 1) {
      Expression expression = ((ExpressionStatement) statement).expressions().get(0);
      if (expression.is(Tree.Kind.CALL_EXPR) && ((CallExpression) expression).callee().is(Tree.Kind.QUALIFIED_EXPR)) {
        QualifiedExpression qualifiedExpression = (QualifiedExpression) ((CallExpression) expression).callee();
        return tryGetName(qualifiedExpression.qualifier()).equals("self")
          && supportedAssertMethods.contains(qualifiedExpression.name().name());
      }
    }

    return false;
  }

  private static String tryGetName(Expression expression) {
    String result = "";

    if (expression.is(Tree.Kind.NAME)) {
      result = ((Name) expression).name();
    }

    return result;
  }
}
