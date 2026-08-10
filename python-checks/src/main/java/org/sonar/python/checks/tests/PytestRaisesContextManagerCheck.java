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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.UnittestUtils;

@Rule(key = "S9000")
public class PytestRaisesContextManagerCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Prefer the context manager form: wrap the raising code in \"with pytest.raises(...)\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> checkCallExpression(ctx, (CallExpression) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkCallExpression(SubscriptionContext ctx, CallExpression callExpression) {
    if (!UnittestUtils.isPytestRaises(callExpression, ctx)) {
      return;
    }
    // Only raise when sure: discarded expression-statement calls, or the deprecated callable form.
    // Assigned / nested / parametrize / returned values are left alone (prefer FN over FP).
    if (isDeprecatedCallableForm(callExpression) || isStandaloneExpressionStatement(callExpression)) {
      ctx.addIssue(callExpression, MESSAGE);
    }
  }

  /**
   * Deprecated {@code pytest.raises(exc, callable, *args, **kwargs)} form always executes immediately
   * and must be migrated to a {@code with} block — safe to report in any syntactic position.
   */
  private static boolean isDeprecatedCallableForm(CallExpression callExpression) {
    return secondPositionalArgument(callExpression.arguments()) != null;
  }

  private static RegularArgument secondPositionalArgument(List<Argument> arguments) {
    int positionalIndex = 0;
    for (Argument argument : arguments) {
      if (!(argument instanceof RegularArgument regularArgument) || regularArgument.keywordArgument() != null) {
        continue;
      }
      if (positionalIndex == 1) {
        return regularArgument;
      }
      positionalIndex++;
    }
    return null;
  }

  /**
   * {@code pytest.raises(...)} used as a bare expression statement discards the context manager —
   * safe to report. Nested uses (assignment RHS, call args, {@code with} items, etc.) are not.
   */
  private static boolean isStandaloneExpressionStatement(CallExpression callExpression) {
    Expression expression = outermostParenthesized(callExpression);
    return expression.parent() instanceof ExpressionStatement;
  }

  private static Expression outermostParenthesized(Expression expression) {
    Expression current = expression;
    while (current.parent() != null && current.parent().is(Tree.Kind.PARENTHESIZED)) {
      current = (Expression) current.parent();
    }
    return current;
  }
}
