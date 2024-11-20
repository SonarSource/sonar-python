/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.checks.tests;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tests.UnittestUtils;

@Rule(key = "S5918")
public class ImplicitlySkippedTestCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Skip this test explicitly.";
  private static final String MESSAGE_RETURN_SKIP = "Remove this return, it is not needed to skip the test.";

  private static final Tree.Kind[] literalsKind = {Tree.Kind.STRING_LITERAL, Tree.Kind.NUMERIC_LITERAL, Tree.Kind.LIST_LITERAL,
    Tree.Kind.BOOLEAN_LITERAL_PATTERN, Tree.Kind.NUMERIC_LITERAL_PATTERN, Tree.Kind.NONE_LITERAL_PATTERN, Tree.Kind.STRING_LITERAL_PATTERN};

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

      if (!functionDef.name().name().startsWith("test")) {
        return;
      }

      ReturnStatement returnStatement = getReturnStatementFromFirstIfStatement(functionDef);
      if (returnStatement == null) {
        return;
      }

      if (hasAnySkipStatement(returnStatement.expressions())) {
        ctx.addIssue(returnStatement.returnKeyword(), MESSAGE_RETURN_SKIP);
        return;
      }

      // We add an issue only if there is an assert statement somewhere in the function definition
      if (containsAssertion(functionDef)) {
        ctx.addIssue(returnStatement, MESSAGE);
      }
    });
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static ReturnStatement getReturnStatementFromFirstIfStatement(FunctionDef functionDef) {
    // check first non assignment statement is an if with a return
    for (Statement statement : functionDef.body().statements()) {
      // skip literal assignment expression
      if (statement.is(Tree.Kind.ASSIGNMENT_STMT) && ((AssignmentStatement) statement).assignedValue().is(literalsKind)) {
        continue;
      }

      if (statement.is(Tree.Kind.IF_STMT)) {
        IfStatement ifStatement = (IfStatement) statement;
        List<Statement> statements = ifStatement.body().statements();
        if (statements.get(0).is(Tree.Kind.RETURN_STMT)) {
          return (ReturnStatement) statements.get(0);
        }
      }

      return null;
    }
    return null;
  }

  private static boolean hasAnySkipStatement(List<Expression> expressions) {
    return expressions.stream()
      .filter(expression -> expression.is(Tree.Kind.CALL_EXPR))
      .map(CallExpression.class::cast)
      .filter(callExpr -> callExpr.callee().is(Tree.Kind.QUALIFIED_EXPR))
      .map(callExpr -> (QualifiedExpression) callExpr.callee())
      .anyMatch(callExpr -> isPytestSkip(callExpr) || isUnittestSkip(callExpr));
  }

  private static boolean isPytestSkip(QualifiedExpression qualifiedExpression) {
    return Optional.of(qualifiedExpression).stream()
      .map(qualifExpr -> qualifExpr.name().symbol())
      .filter(Objects::nonNull)
      .anyMatch(symbol -> "pytest.skip".equals(symbol.fullyQualifiedName()));
  }

  private static boolean isUnittestSkip(QualifiedExpression qualifiedExpression) {
    return Optional.of(qualifiedExpression).stream()
      .anyMatch(qualifExpr -> qualifExpr.qualifier().is(Tree.Kind.NAME) && "self".equals(((Name) qualifExpr.qualifier()).name())
        && "skipTest".equals(qualifExpr.name().name()));
  }

  private static boolean containsAssertion(FunctionDef functionDef) {
    Set<String> supportedAssertMethods = UnittestUtils.isWithinUnittestTestCase(functionDef) ?
      UnittestUtils.allAssertMethods() : Collections.emptySet();
    AssertionVisitor assertVisitor = new AssertionVisitor(supportedAssertMethods);
    functionDef.accept(assertVisitor);
    return assertVisitor.hasAnAssert;
  }

  static class AssertionVisitor extends BaseTreeVisitor {
    boolean hasAnAssert = false;
    Set<String> supportedMethods;

    public AssertionVisitor(Set<String> supportedMethods) {
      this.supportedMethods = supportedMethods;
    }

    @Override
    public void visitAssertStatement(AssertStatement assertStatement) {
      hasAnAssert = true;
      super.visitAssertStatement(assertStatement);
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
        QualifiedExpression qualifiedExpression = (QualifiedExpression) callExpression.callee();
        if (qualifiedExpression.qualifier().is(Tree.Kind.NAME) && "self".equals(((Name) qualifiedExpression.qualifier()).name())
          && supportedMethods.contains(qualifiedExpression.name().name())) {
          hasAnAssert = true;
        }
      }
      super.visitCallExpression(callExpression);
    }
  }
}
