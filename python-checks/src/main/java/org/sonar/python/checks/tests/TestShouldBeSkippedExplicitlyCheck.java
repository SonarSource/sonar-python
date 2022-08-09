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
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
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
import org.sonar.python.checks.FunctionUsingLoopVariableCheck;

@Rule(key = "S5918")
public class TestShouldBeSkippedExplicitlyCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Skip this test explicitly.";
  private static final Set<String> supportedAssertMethods = new HashSet<>(Arrays.asList("assertEqual", "assertNotEqual", "assertTrue",
    "assertFalse", "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn", "assertNotIn", "assertIsInstance",
    "assertNotIsInstance", "assertRaises", "assertRaisesRegexp", "assertRaisesRegex", "assertAlmostEqual", "assertNotAlmostEqual",
    "assertGreater", "assertGreaterEqual", "assertLess", "assertLessEqual", "assertRegexpMatches", "assertNotRegexpMatches",
    "assertItemsEqual", "assertDictContainsSubset"));
  private static final Tree.Kind[] literalsKind = {Tree.Kind.STRING_LITERAL, Tree.Kind.NUMERIC_LITERAL, Tree.Kind.LIST_LITERAL,
    Tree.Kind.BOOLEAN_LITERAL_PATTERN, Tree.Kind.NUMERIC_LITERAL_PATTERN, Tree.Kind.NONE_LITERAL_PATTERN, Tree.Kind.STRING_LITERAL_PATTERN,
    Tree.Kind.SET_LITERAL, Tree.Kind.DICTIONARY_LITERAL};

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

      if (!functionDef.name().name().startsWith("test")) {
        return;
      }

      ReturnStatement returnStatement = getReturnStatementInIfFirstStatement(functionDef);
      if (returnStatement == null) {
        return;
      }

      // check there is an assert like statement in the function
      if (isOrContainAssertStatement(functionDef)) {
        ctx.addIssue(returnStatement, MESSAGE);
      }
    });
  }

  private static ReturnStatement getReturnStatementInIfFirstStatement(FunctionDef functionDef) {
    // check first non assignment statement is an if with a return
    for (Statement statement : functionDef.body().statements()) {
      // skip literal assignment expression
      if (statement.is(Tree.Kind.ASSIGNMENT_STMT) && ((AssignmentStatement) statement).assignedValue().is(literalsKind)) {
        continue;
      }

      if (statement.is(Tree.Kind.IF_STMT)) {
        IfStatement ifStatement = (IfStatement) statement;
        List<Statement> statements = ifStatement.body().statements();
        if (!statements.isEmpty() && statements.get(0).is(Tree.Kind.RETURN_STMT)) {
          return (ReturnStatement) statements.get(0);
        }
      }

      return null;
    }
    return null;
  }

  private static boolean isOrContainAssertStatement(Tree node) {
    AssertOrCallAssertVisitor assertOrCallAssertVisitor = new AssertOrCallAssertVisitor();
    node.accept(assertOrCallAssertVisitor);
    return assertOrCallAssertVisitor.hasACallToAssert || assertOrCallAssertVisitor.hasANativeAssert;
  }

  static class AssertOrCallAssertVisitor extends BaseTreeVisitor {
    boolean hasANativeAssert = false;
    boolean hasACallToAssert = false;

    @Override
    public void visitAssertStatement(AssertStatement pyAssertStatementTree) {
      hasANativeAssert = true;
      super.visitAssertStatement(pyAssertStatementTree);
    }

    @Override
    public void visitQualifiedExpression(QualifiedExpression pyQualifiedExpressionTree) {
      if (tryGetName(pyQualifiedExpressionTree.qualifier()).equals("self")
        && supportedAssertMethods.contains(pyQualifiedExpressionTree.name().name())) {
        hasACallToAssert = true;
      }
      super.visitQualifiedExpression(pyQualifiedExpressionTree);
    }

    private static String tryGetName(Expression expression) {
      String result = "";

      if (expression.is(Tree.Kind.NAME)) {
        result = ((Name) expression).name();
      }

      return result;
    }
  }

}
