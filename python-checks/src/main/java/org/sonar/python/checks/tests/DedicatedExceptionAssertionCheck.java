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
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tests.UnittestUtils;

@Rule(key = "S8714")
public class DedicatedExceptionAssertionCheck extends PythonSubscriptionCheck {
  private static final String PYTEST_FAIL_FQN = "pytest.fail";
  private static final String PYTEST_MESSAGE = "Replace this try/except block with a \"pytest.raises\" context manager.";
  private static final String UNITTEST_MESSAGE = "Replace this try/except block with \"self.assertRaises()\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TRY_STMT, ctx -> checkTryStatement(ctx, (TryStatement) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkTryStatement(SubscriptionContext ctx, TryStatement tryStatement) {
    if (!hasSimpleExceptClauses(tryStatement) || tryStatement.finallyClause() != null) {
      return;
    }

    CallExpression failCall = failCallFromElseClause(tryStatement.elseClause());
    if (failCall == null) {
      failCall = failCallFromTryBody(tryStatement.body().statements());
    }
    if (failCall == null) {
      return;
    }

    String message = issueMessage(failCall);
    if (message != null) {
      ctx.addIssue(failCall, message);
    }
  }

  private static boolean hasSimpleExceptClauses(TryStatement tryStatement) {
    return !tryStatement.exceptClauses().isEmpty() && tryStatement.exceptClauses().stream().allMatch(exceptClause ->
      exceptClause.starToken() == null
        && exceptClause.exception() != null
        && exceptClause.body().statements().stream().allMatch(CheckUtils::isEmptyStatement));
  }

  @Nullable
  private static CallExpression failCallFromElseClause(@Nullable ElseClause elseClause) {
    if (elseClause == null) {
      return null;
    }
    List<Statement> statements = elseClause.body().statements();
    if (statements.size() != 1) {
      return null;
    }
    return failCallFromStatement(statements.get(0));
  }

  @Nullable
  private static CallExpression failCallFromTryBody(List<Statement> statements) {
    if (statements.size() < 2) {
      return null;
    }
    return failCallFromStatement(statements.get(statements.size() - 1));
  }

  @Nullable
  private static CallExpression failCallFromStatement(Statement statement) {
    if (!(statement instanceof ExpressionStatement expressionStatement) || expressionStatement.expressions().size() != 1) {
      return null;
    }
    Expression expression = expressionStatement.expressions().get(0);
    if (!(expression instanceof CallExpression callExpression)) {
      return null;
    }
    return isPytestFail(callExpression) || isUnittestFail(callExpression) ? callExpression : null;
  }

  private static boolean isPytestFail(CallExpression callExpression) {
    Symbol calleeSymbol = callExpression.calleeSymbol();
    return calleeSymbol != null && PYTEST_FAIL_FQN.equals(calleeSymbol.fullyQualifiedName());
  }

  private static boolean isUnittestFail(CallExpression callExpression) {
    if (!UnittestUtils.isWithinUnittestTestCase(callExpression)) {
      return false;
    }
    Expression callee = callExpression.callee();
    if (!(callee instanceof QualifiedExpression qualifiedExpression)) {
      return false;
    }
    if (!(qualifiedExpression.qualifier() instanceof Name qualifier) || !"self".equals(qualifier.name())) {
      return false;
    }
    return "fail".equals(qualifiedExpression.name().name());
  }

  @Nullable
  private static String issueMessage(CallExpression failCall) {
    if (isPytestFail(failCall)) {
      return PYTEST_MESSAGE;
    }
    if (isUnittestFail(failCall)) {
      return UNITTEST_MESSAGE;
    }
    return null;
  }
}
