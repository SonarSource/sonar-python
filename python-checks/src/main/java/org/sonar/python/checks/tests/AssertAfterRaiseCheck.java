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

import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithStatement;

@Rule(key = "S5915")
public class AssertAfterRaiseCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE_MULTIPLE_STATEMENT = "Don’t perform an assertion here; An exception is expected to be raised before its execution.";
  private static final String MESSAGE_SINGLE_STATEMENT = "Refactor this test; if this assertion’s argument raises an exception, the assertion will never get executed.";
  private static final String MESSAGE_SECONDARY = "test";
  private static final Set<String> raiseStatements = Set.of("pytest.raises", "self.assertRaises");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, ctx -> {
      WithStatement withStatement = (WithStatement) ctx.syntaxNode();
      if (!isWithStatementARaise(withStatement)) {
        return;
      }

      List<Statement> statements = withStatement.statements().statements();
      if (statements.isEmpty()) {
        return;
      }

      Statement statement = statements.get(statements.size()-1);
      if (isAnAssert(statement)) {
        ctx.addIssue(statement, statements.size() > 1 ? MESSAGE_MULTIPLE_STATEMENT : MESSAGE_SINGLE_STATEMENT)
          .secondary(IssueLocation.preciseLocation(withStatement.firstToken(), withStatement.colon(), MESSAGE_SECONDARY));
      }
    });
  }

  public boolean isWithStatementARaise(WithStatement withStatement) {
    return withStatement.withItems().stream()
      .filter(withItem -> withItem.test().is(Tree.Kind.CALL_EXPR))
      .map(withItem -> ((CallExpression) withItem.test()).callee())
      .filter(callee -> callee.is(Tree.Kind.QUALIFIED_EXPR))
      .map(QualifiedExpression.class::cast)
      .filter(PytestUtils::isARaiseCall) // TODO : add UnittestUtils call
      .filter(callee -> callee.qualifier().is(Tree.Kind.NAME))
      .map(callee -> ((Name) callee.qualifier()).name() + "." + callee.name().name())
      .anyMatch(raiseStatements::contains);
  }

  public boolean isAnAssert(Statement statement) {
    if (statement.is(Tree.Kind.ASSERT_STMT)) {
      return true;
    }

    return Optional.of(statement).stream()
      .filter(stat -> stat.is(Tree.Kind.EXPRESSION_STMT))
      .map(ExpressionStatement.class::cast)
      .map(ExpressionStatement::expressions)
      .anyMatch(expressions ->
        expressions.stream()
          .filter(expression -> expression.is(Tree.Kind.CALL_EXPR))
          .map(CallExpression.class::cast)
          .anyMatch(TestFrameworkUtils::isAnAssertOfAnySupportedTestFramework)
      );
  }
}
