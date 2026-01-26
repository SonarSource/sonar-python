/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8371")
public class FlaskHeadersDictAccessCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use \".get()\" method to safely access this header.";

  // We flag both Flask and direct Werkzeug usage because Flask uses Werkzeug's Headers under the hood.
  private static final TypeMatcher WERKZEUG_HEADERS_MATCHER = TypeMatchers.isObjectInstanceOf("werkzeug.datastructures.headers.Headers");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.SUBSCRIPTION, FlaskHeadersDictAccessCheck::checkSubscription);
  }

  private static void checkSubscription(SubscriptionContext ctx) {
    SubscriptionExpression subscriptionExpression = (SubscriptionExpression) ctx.syntaxNode();

    if (isAssignmentTarget(subscriptionExpression)) {
      return;
    }

    if (WERKZEUG_HEADERS_MATCHER.isTrueFor(subscriptionExpression.object(), ctx)) {
      ctx.addIssue(subscriptionExpression, MESSAGE);
    }
  }

  private static boolean isAssignmentTarget(SubscriptionExpression subscriptionExpression) {
    return TreeUtils.firstAncestor(subscriptionExpression, t -> t.is(Tree.Kind.ASSIGNMENT_STMT)
      && ((AssignmentStatement) t).lhsExpressions().stream()
        .flatMap(lhs -> lhs.expressions().stream())
        .anyMatch(expr -> expr.equals(subscriptionExpression))) != null;
  }
}
