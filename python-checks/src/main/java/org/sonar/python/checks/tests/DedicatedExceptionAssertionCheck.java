/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.List;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S8714")
public class DedicatedExceptionAssertionCheck extends PythonSubscriptionCheck {
  private static final TypeMatcher PYTEST_FAIL_MATCHER = TypeMatchers.withFQN("pytest.fail");
  private static final TypeMatcher UNITTEST_FAIL_MATCHER = TypeMatchers.withFQN("unittest.case.TestCase.fail");
  private static final String PYTEST_MESSAGE = "Replace this try/except block with a \"pytest.raises\" context manager.";
  private static final String UNITTEST_MESSAGE = "Replace this try/except block with \"self.assertRaises()\".";
  private static final String NO_EXCEPTION_MESSAGE = "Remove this try/except block and let the test fail naturally if an exception is raised.";
  private static final String DEDICATED_ASSERTION_SECONDARY_MESSAGE = "Replace this fail call with a dedicated exception assertion.";
  private static final String NO_EXCEPTION_SECONDARY_MESSAGE = "Remove this fail call and let the test fail naturally if an exception is raised.";

  private record FailCall(CallExpression callExpression, String primaryMessage, String secondaryMessage) {
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TRY_STMT, ctx -> checkTryStatement(ctx, (TryStatement) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.TESTS;
  }

  private static void checkTryStatement(SubscriptionContext ctx, TryStatement tryStatement) {
    if (!hasSupportedExceptClauses(tryStatement) || tryStatement.finallyClause() != null) {
      return;
    }

    if (tryStatement.exceptClauses().size() == 1) {
      FailCall exceptFail = failCallFromSingleStatementBody(tryStatement.exceptClauses().get(0).body().statements(), ctx);
      if (exceptFail != null) {
        ctx.addIssue(tryStatement.tryKeyword(), endToken(tryStatement), NO_EXCEPTION_MESSAGE)
          .secondary(exceptFail.callExpression(), NO_EXCEPTION_SECONDARY_MESSAGE);
        return;
      }
    }

    FailCall failCall = failCallFromElseClause(tryStatement.elseClause(), ctx);
    if (failCall == null) {
      failCall = failCallFromTryBody(tryStatement.body().statements(), ctx);
    }
    if (failCall == null) {
      return;
    }

    ctx.addIssue(tryStatement.tryKeyword(), endToken(tryStatement), failCall.primaryMessage())
      .secondary(failCall.callExpression(), failCall.secondaryMessage());
  }

  private static Token endToken(TryStatement tryStatement) {
    var elseClause = tryStatement.elseClause();
    if (elseClause != null) {
      return lastStatement(elseClause.body().statements()).lastToken();
    }
    return lastStatement(tryStatement.exceptClauses().get(tryStatement.exceptClauses().size() - 1).body().statements()).lastToken();
  }

  private static Statement lastStatement(List<Statement> statements) {
    return statements.get(statements.size() - 1);
  }

  private static boolean hasSupportedExceptClauses(TryStatement tryStatement) {
    return !tryStatement.exceptClauses().isEmpty() && tryStatement.exceptClauses().stream().allMatch(exceptClause ->
      exceptClause.starToken() == null && exceptClause.exception() != null);
  }

  @Nullable
  private static FailCall failCallFromElseClause(@Nullable ElseClause elseClause, SubscriptionContext ctx) {
    if (elseClause == null) {
      return null;
    }
    return failCallFromSingleStatementBody(elseClause.body().statements(), ctx);
  }

  @Nullable
  private static FailCall failCallFromTryBody(List<Statement> statements, SubscriptionContext ctx) {
    if (statements.size() < 2) {
      return null;
    }
    return failCallFromStatement(statements.get(statements.size() - 1), ctx);
  }

  @Nullable
  private static FailCall failCallFromSingleStatementBody(List<Statement> statements, SubscriptionContext ctx) {
    if (statements.size() != 1) {
      return null;
    }
    return failCallFromStatement(statements.get(0), ctx);
  }

  @Nullable
  private static FailCall failCallFromStatement(Statement statement, SubscriptionContext ctx) {
    if (!(statement instanceof ExpressionStatement expressionStatement) || expressionStatement.expressions().size() != 1) {
      return null;
    }
    Expression expression = expressionStatement.expressions().get(0);
    if (!(expression instanceof CallExpression callExpression)) {
      return null;
    }
    if (isPytestFail(callExpression, ctx)) {
      return new FailCall(callExpression, PYTEST_MESSAGE, DEDICATED_ASSERTION_SECONDARY_MESSAGE);
    }
    if (isUnittestFail(callExpression, ctx)) {
      return new FailCall(callExpression, UNITTEST_MESSAGE, DEDICATED_ASSERTION_SECONDARY_MESSAGE);
    }
    return null;
  }

  private static boolean isPytestFail(CallExpression callExpression, SubscriptionContext ctx) {
    return PYTEST_FAIL_MATCHER.isTrueFor(callExpression.callee(), ctx);
  }

  private static boolean isUnittestFail(CallExpression callExpression, SubscriptionContext ctx) {
    return UNITTEST_FAIL_MATCHER.isTrueFor(callExpression.callee(), ctx);
  }
}
