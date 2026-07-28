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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;

@Rule(key = "S9075")
public class SpecificWarningAssertionCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "This assertion is too broad; use a more specific warning type or check the warning message.";

  private static final TypeMatcher BROAD_WARNING_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("Warning"),
    TypeMatchers.isObjectOfType("Warning")
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> checkCallExpression(ctx, (CallExpression) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkCallExpression(SubscriptionContext ctx, CallExpression callExpression) {
    Expression issueLocation = underspecifiedWarningArgument(callExpression, ctx);
    if (issueLocation != null) {
      ctx.addIssue(issueLocation, MESSAGE);
    }
  }

  @Nullable
  private static Expression underspecifiedWarningArgument(CallExpression callExpression, SubscriptionContext ctx) {
    if (!UnittestUtils.isPytestWarns(callExpression, ctx)) {
      return null;
    }
    RegularArgument warningArgument = UnittestUtils.pytestExpectedWarningArgument(callExpression);
    if (warningArgument == null) {
      return callExpression;
    }
    if (hasEffectiveMatchArgument(callExpression)) {
      return null;
    }
    return broadWarningLocation(warningArgument.expression(), ctx);
  }

  @Nullable
  private static Expression broadWarningLocation(Expression expression, SubscriptionContext ctx) {
    Expression unwrapped = Expressions.removeParentheses(expression);
    List<Expression> elements = Expressions.expressionsFromListOrTuple(unwrapped);
    if (!elements.isEmpty()) {
      for (Expression element : elements) {
        Expression location = broadWarningLocation(element, ctx);
        if (location != null) {
          return location;
        }
      }
      return null;
    }
    if (BROAD_WARNING_MATCHER.isTrueFor(unwrapped, ctx)) {
      return unwrapped;
    }
    return null;
  }

  private static boolean hasEffectiveMatchArgument(CallExpression callExpression) {
    RegularArgument matchArgument = UnittestUtils.pytestMatchArgument(callExpression);
    return matchArgument != null && !Expressions.isFalsy(matchArgument.expression());
  }
}
