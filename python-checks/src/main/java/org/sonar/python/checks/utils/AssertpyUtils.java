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
package org.sonar.python.checks.utils;

import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

public final class AssertpyUtils {

  private static final TypeMatcher ASSERTPY_ASSERT_THAT_MATCHER = TypeMatchers.isType("assertpy.assert_that");

  private AssertpyUtils() {
  }

  public static boolean isAssertThatCall(CallExpression callExpression, SubscriptionContext ctx) {
    return ASSERTPY_ASSERT_THAT_MATCHER.isTrueFor(callExpression.callee(), ctx);
  }

  @Nullable
  public static CallExpression originatingAssertThatCall(Expression expression, SubscriptionContext ctx) {
    Expression qualifier = Expressions.removeParentheses(expression);
    while (qualifier instanceof CallExpression callExpression) {
      if (isAssertThatCall(callExpression, ctx)) {
        return callExpression;
      }
      Expression callee = Expressions.removeParentheses(callExpression.callee());
      if (!(callee instanceof QualifiedExpression qualifiedExpression)) {
        return null;
      }
      qualifier = Expressions.removeParentheses(qualifiedExpression.qualifier());
    }
    return null;
  }
}
