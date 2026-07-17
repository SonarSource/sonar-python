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

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9001")
public class XfailNoReasonCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Provide a reason for marking this test as expected to fail.";

  private static final TypeMatcher PYTEST_XFAIL_MATCHER = TypeMatchers.isType("pytest.mark.xfail");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, ctx -> checkDecorator(ctx, (Decorator) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkDecorator(SubscriptionContext ctx, Decorator decorator) {
    Expression expression = decorator.expression();
    Expression xfailExpression = getXfailExpression(expression);
    if (!PYTEST_XFAIL_MATCHER.isTrueFor(xfailExpression, ctx)) {
      return;
    }

    checkMissingReason(ctx, decorator, getArguments(decorator, expression));
  }

  private static Expression getXfailExpression(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      return ((CallExpression) expression).callee();
    }
    return expression;
  }

  @Nullable
  private static ArgList getArguments(Decorator decorator, Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      return ((CallExpression) expression).argumentList();
    }
    return decorator.arguments();
  }

  private static void checkMissingReason(SubscriptionContext ctx, Tree issueTree, @Nullable ArgList args) {
    if (args == null || args.arguments().isEmpty()) {
      ctx.addIssue(issueTree, MESSAGE);
      return;
    }

    RegularArgument reasonArgument = TreeUtils.argumentByKeyword("reason", args.arguments());
    if (reasonArgument == null) {
      ctx.addIssue(issueTree, MESSAGE);
      return;
    }

    if (reasonArgument.expression().is(Tree.Kind.STRING_LITERAL)) {
      StringLiteral stringLiteral = (StringLiteral) reasonArgument.expression();
      if (stringLiteral.trimmedQuotesValue().isEmpty()) {
        ctx.addIssue(issueTree, MESSAGE);
      }
    }
  }
}
