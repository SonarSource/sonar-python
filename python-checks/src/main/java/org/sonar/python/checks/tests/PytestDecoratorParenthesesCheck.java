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

import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S9083")
public class PytestDecoratorParenthesesCheck extends PythonSubscriptionCheck {

  private static final boolean DEFAULT_REQUIRE_PARENTHESES = false;

  private static final String REMOVE_MESSAGE = "Remove empty parentheses from this decorator.";
  private static final String ADD_MESSAGE = "Add empty parentheses to this decorator.";
  private static final String REMOVE_QUICK_FIX_MESSAGE = "Remove the empty parentheses";
  private static final String ADD_QUICK_FIX_MESSAGE = "Add empty parentheses";

  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN("pytest.fixture");
  private static final TypeMatcher PYTEST_MARK_MATCHER = TypeMatchers.isType("pytest.mark");
  private static final TypeMatcher KNOWN_PYTEST_MARK_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("pytest.mark.skip"),
    TypeMatchers.isType("pytest.mark.xfail"),
    TypeMatchers.isType("pytest.mark.parametrize"),
    TypeMatchers.isType("pytest.mark.usefixtures"));

  @RuleProperty(
    key = "requireParentheses",
    description = "Whether argument-free pytest fixture and mark decorators should use empty parentheses.",
    defaultValue = "" + DEFAULT_REQUIRE_PARENTHESES)
  public boolean requireParentheses = DEFAULT_REQUIRE_PARENTHESES;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, this::checkDecorator);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private void checkDecorator(SubscriptionContext ctx) {
    Expression expression = ((Decorator) ctx.syntaxNode()).expression();
    if (expression instanceof CallExpression callExpression) {
      if (!requireParentheses && callExpression.arguments().isEmpty() && isPytestFixtureOrMark(callExpression.callee(), ctx)) {
        raiseOnEmptyParentheses(ctx, callExpression);
      }
    } else if (requireParentheses && isPytestFixtureOrMark(expression, ctx)) {
      raiseOnMissingParentheses(ctx, expression);
    }
  }

  private static void raiseOnEmptyParentheses(SubscriptionContext ctx, CallExpression callExpression) {
    Token rightPar = callExpression.rightPar();
    var issue = ctx.addIssue(callExpression.leftPar(), rightPar, REMOVE_MESSAGE);
    // A comment between the parentheses would be dropped along with them, so no fix is offered in that case.
    if (rightPar.trivia().isEmpty()) {
      Token calleeEnd = callExpression.callee().lastToken();
      issue.addQuickFix(PythonQuickFix.newQuickFix(REMOVE_QUICK_FIX_MESSAGE)
        .addTextEdit(TextEditUtils.removeRange(calleeEnd.pythonLine(), endColumn(calleeEnd), rightPar.pythonLine(), endColumn(rightPar)))
        .build());
    }
  }

  private static void raiseOnMissingParentheses(SubscriptionContext ctx, Expression expression) {
    ctx.addIssue(expression, ADD_MESSAGE)
      .addQuickFix(PythonQuickFix.newQuickFix(ADD_QUICK_FIX_MESSAGE)
        .addTextEdit(TextEditUtils.insertAfter(expression.lastToken(), "()"))
        .build());
  }

  private static boolean isPytestFixtureOrMark(Expression expression, SubscriptionContext ctx) {
    if (PYTEST_FIXTURE_MATCHER.isTrueFor(expression, ctx) || KNOWN_PYTEST_MARK_MATCHER.isTrueFor(expression, ctx)) {
      return true;
    }
    // Custom marks have no stub, so they are recognized through the `pytest.mark` qualifier they are accessed on.
    return expression instanceof QualifiedExpression qualifiedExpression
      && PYTEST_MARK_MATCHER.isTrueFor(qualifiedExpression.qualifier(), ctx);
  }

  private static int endColumn(Token token) {
    return token.pythonColumn() + token.value().length();
  }
}
