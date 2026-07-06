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
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.UnittestUtils;

import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.EXCEPTION;

@Rule(key = "S5958")
public class SpecificExceptionAssertionCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Specify a more specific exception type here.";
  private static final TypeMatcher GENERIC_EXCEPTION_MATCHER = TypeMatchers.any(
    TypeMatchers.isObjectOfType(EXCEPTION),
    TypeMatchers.isObjectOfType(BASE_EXCEPTION),
    TypeMatchers.isType(EXCEPTION),
    TypeMatchers.isType(BASE_EXCEPTION)
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, ctx -> checkWithStatement(ctx, (WithStatement) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> checkCallExpression(ctx, (CallExpression) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkWithStatement(SubscriptionContext ctx, WithStatement withStatement) {
    for (WithItem withItem : withStatement.withItems()) {
      if (!(withItem.test() instanceof CallExpression callExpression)) {
        continue;
      }
      Expression issueLocation = genericExceptionArgument(callExpression, ctx);
      if (issueLocation != null) {
        ctx.addIssue(issueLocation, MESSAGE);
      }
    }
  }

  private static void checkCallExpression(SubscriptionContext ctx, CallExpression callExpression) {
    if (callExpression.parent().is(Tree.Kind.WITH_ITEM)) {
      return;
    }
    Expression issueLocation = genericExceptionArgument(callExpression, ctx);
    if (issueLocation != null) {
      ctx.addIssue(issueLocation, MESSAGE);
    }
  }

  @Nullable
  private static Expression genericExceptionArgument(CallExpression callExpression, SubscriptionContext ctx) {
    RegularArgument exceptionArgument = null;
    if (UnittestUtils.isPytestRaises(callExpression, ctx)) {
      if (UnittestUtils.hasPytestRaisesMatchArgument(callExpression)) {
        return null;
      }
      exceptionArgument = UnittestUtils.pytestExpectedExceptionArgument(callExpression);
    } else if (UnittestUtils.isUnittestAssertRaises(callExpression, ctx)) {
      if (UnittestUtils.hasUnittestAssertRaisesMessageCheck(callExpression, ctx)) {
        return null;
      }
      exceptionArgument = UnittestUtils.unittestExceptionArgument(callExpression);
    }
    if (exceptionArgument == null || !GENERIC_EXCEPTION_MATCHER.isTrueFor(exceptionArgument.expression(), ctx)) {
      return null;
    }
    return exceptionArgument.expression();
  }
}
