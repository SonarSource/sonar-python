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
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.AssertpyUtils;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.EXCEPTION;

@Rule(key = "S5779")
public class AssertionInTryCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Don't use %s inside a try-except that catches AssertionError.";
  private static final String ASSERTION_ERROR_SECONDARY_MESSAGE = "AssertionError is caught here.";
  private static final String EXCEPTION_SECONDARY_MESSAGE = "Exception is caught here.";
  private static final String BASE_EXCEPTION_SECONDARY_MESSAGE = "BaseException is caught here.";
  private static final String BARE_EXCEPT_SECONDARY_MESSAGE = "All exceptions are caught here.";

  private static final TypeMatcher ASSERTION_ERROR_TYPE_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("AssertionError"),
    TypeMatchers.isOrExtendsType("builtins.AssertionError"));
  private static final TypeMatcher EXCEPTION_TYPE_MATCHER = TypeMatchers.any(
    TypeMatchers.isObjectOfType(EXCEPTION),
    TypeMatchers.isType(EXCEPTION));
  private static final TypeMatcher BASE_EXCEPTION_TYPE_MATCHER = TypeMatchers.any(
    TypeMatchers.isObjectOfType(BASE_EXCEPTION),
    TypeMatchers.isType(BASE_EXCEPTION));
  private static final TypeMatcher CATCHES_ASSERTION_ERROR_MATCHER = TypeMatchers.any(
    ASSERTION_ERROR_TYPE_MATCHER,
    EXCEPTION_TYPE_MATCHER,
    BASE_EXCEPTION_TYPE_MATCHER);
  private static final List<Map.Entry<TypeMatcher, String>> SECONDARY_MESSAGES_BY_EXCEPTION_TYPE = List.of(
    Map.entry(ASSERTION_ERROR_TYPE_MATCHER, ASSERTION_ERROR_SECONDARY_MESSAGE),
    Map.entry(EXCEPTION_TYPE_MATCHER, EXCEPTION_SECONDARY_MESSAGE),
    Map.entry(BASE_EXCEPTION_TYPE_MATCHER, BASE_EXCEPTION_SECONDARY_MESSAGE));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> checkAssertion(ctx, ctx.syntaxNode(), "assert"));
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> checkCallExpression(ctx, (CallExpression) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkCallExpression(SubscriptionContext ctx, CallExpression callExpression) {
    String assertionName = unittestAssertionName(callExpression);
    if (assertionName == null) {
      assertionName = assertpyAssertionName(callExpression, ctx);
    }
    if (assertionName != null) {
      checkAssertion(ctx, callExpression, assertionName);
    }
  }

  private static void checkAssertion(SubscriptionContext ctx, Tree assertion, String assertionName) {
    findTrySwallowingAssertionError(assertion, ctx).ifPresent(tryStatement -> {
      ExceptClause exceptClause = exceptClauseCatchingAssertionError(tryStatement, ctx);
      ctx.addIssue(assertion, String.format(MESSAGE, assertionName))
        .secondary(exceptClause.exception() != null ? exceptClause.exception() : exceptClause.exceptKeyword(),
          secondaryMessage(exceptClause, ctx));
    });
  }

  private static Optional<TryStatement> findTrySwallowingAssertionError(Tree tree, SubscriptionContext ctx) {
    TryStatement tryStatement = enclosingTryStatementInBody(tree);
    while (tryStatement != null) {
      ExceptClause exceptClause = exceptClauseCatchingAssertionError(tryStatement, ctx);
      if (exceptClause != null && !exceptClauseReraises(exceptClause)) {
        return Optional.of(tryStatement);
      }
      tryStatement = enclosingTryStatementInBody(tryStatement);
    }
    return Optional.empty();
  }

  @Nullable
  private static TryStatement enclosingTryStatementInBody(Tree tree) {
    TryStatement tryStatement = (TryStatement) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.TRY_STMT);
    while (tryStatement != null) {
      if (isDescendantOf(tryStatement.body(), tree)) {
        return tryStatement;
      }
      tryStatement = (TryStatement) TreeUtils.firstAncestorOfKind(tryStatement.parent(), Tree.Kind.TRY_STMT);
    }
    return null;
  }

  @Nullable
  private static ExceptClause exceptClauseCatchingAssertionError(TryStatement tryStatement, SubscriptionContext ctx) {
    for (ExceptClause exceptClause : tryStatement.exceptClauses()) {
      if (exceptClause.starToken() != null) {
        continue;
      }
      Expression exception = exceptClause.exception();
      if (exception == null) {
        return exceptClause;
      }
      for (Expression caughtException : TreeUtils.flattenTuples(exception).toList()) {
        if (canCatchAssertionError(caughtException, ctx)) {
          return exceptClause;
        }
      }
    }
    return null;
  }

  private static boolean canCatchAssertionError(Expression caughtException, SubscriptionContext ctx) {
    return CATCHES_ASSERTION_ERROR_MATCHER.isTrueFor(Expressions.removeParentheses(caughtException), ctx);
  }

  private static boolean exceptClauseReraises(ExceptClause exceptClause) {
    return TreeUtils.hasDescendant(exceptClause.body(), tree -> {
      if (!tree.is(Tree.Kind.RAISE_STMT)) {
        return false;
      }
      RaiseStatement raiseStatement = (RaiseStatement) tree;
      if (raiseStatement.expressions().isEmpty()) {
        return true;
      }
      if (exceptClause.exceptionInstance() == null || raiseStatement.expressions().size() != 1) {
        return false;
      }
      return CheckUtils.areEquivalent(
        Expressions.removeParentheses(exceptClause.exceptionInstance()),
        Expressions.removeParentheses(raiseStatement.expressions().get(0)));
    });
  }

  private static String secondaryMessage(ExceptClause exceptClause, SubscriptionContext ctx) {
    Expression exception = exceptClause.exception();
    if (exception == null) {
      return BARE_EXCEPT_SECONDARY_MESSAGE;
    }
    for (Expression caughtException : TreeUtils.flattenTuples(exception).toList()) {
      String message = secondaryMessageForException(Expressions.removeParentheses(caughtException), ctx);
      if (message != null) {
        return message;
      }
    }
    return ASSERTION_ERROR_SECONDARY_MESSAGE;
  }

  @Nullable
  private static String secondaryMessageForException(Expression caughtException, SubscriptionContext ctx) {
    return SECONDARY_MESSAGES_BY_EXCEPTION_TYPE.stream()
      .filter(entry -> entry.getKey().isTrueFor(caughtException, ctx))
      .map(Map.Entry::getValue)
      .findFirst()
      .orElse(null);
  }

  @Nullable
  private static String unittestAssertionName(CallExpression callExpression) {
    ExpressionStatement parent = parentExpressionStatement(callExpression);
    if (parent == null || parent.expressions().size() != 1) {
      return null;
    }
    Expression callee = Expressions.removeParentheses(callExpression.callee());
    if (!(callee instanceof QualifiedExpression qualifiedExpression)) {
      return null;
    }
    if (!(qualifiedExpression.qualifier() instanceof Name qualifier) || !"self".equals(qualifier.name())) {
      return null;
    }
    String methodName = qualifiedExpression.name().name();
    return UnittestUtils.allAssertMethods().contains(methodName) ? methodName : null;
  }

  @Nullable
  private static String assertpyAssertionName(CallExpression callExpression, SubscriptionContext ctx) {
    ExpressionStatement parent = parentExpressionStatement(callExpression);
    if (parent == null) {
      return null;
    }
    if (AssertpyUtils.isAssertThatCall(callExpression, ctx)) {
      if (callExpression.parent().is(Tree.Kind.QUALIFIED_EXPR)) {
        return null;
      }
      return "assert_that";
    }
    Expression callee = Expressions.removeParentheses(callExpression.callee());
    if (!(callee instanceof QualifiedExpression qualifiedExpression)) {
      return null;
    }
    if (AssertpyUtils.originatingAssertThatCall(qualifiedExpression.qualifier(), ctx) == null) {
      return null;
    }
    return qualifiedExpression.name().name();
  }

  @Nullable
  private static ExpressionStatement parentExpressionStatement(CallExpression callExpression) {
    if (!(callExpression.parent() instanceof ExpressionStatement expressionStatement)) {
      return null;
    }
    return expressionStatement;
  }

  private static boolean isDescendantOf(StatementList statementList, Tree tree) {
    return TreeUtils.hasDescendant(statementList, descendant -> descendant == tree);
  }
}
