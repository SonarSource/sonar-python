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

import java.util.Set;
import javax.annotation.CheckForNull;
import org.apache.commons.text.similarity.LevenshteinDistance;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S9136")
public class MisspelledMockAssertionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Correct this misspelled mock assertion; did you mean \"%s\"?";
  private static final String QUICK_FIX_MESSAGE = "Replace with \"%s\"";

  /**
   * Only raise when the accessed name is a near-miss of a real mock assertion API.
   * Distance 2 still requires a unique best match, which keeps the rule conservative.
   */
  private static final int MAX_EDIT_DISTANCE = 2;
  private static final LevenshteinDistance LEVENSHTEIN = new LevenshteinDistance(MAX_EDIT_DISTANCE);

  private static final Set<String> KNOWN_ASSERTION_METHODS = Set.of(
    "assert_called",
    "assert_called_once",
    "assert_called_with",
    "assert_called_once_with",
    "assert_any_call",
    "assert_has_calls",
    "assert_not_called",
    "assert_awaited",
    "assert_awaited_once",
    "assert_awaited_with",
    "assert_awaited_once_with",
    "assert_any_await",
    "assert_has_awaits",
    "assert_not_awaited");

  private static final TypeMatcher MOCK_INSTANCE = TypeMatchers.any(
    TypeMatchers.isObjectInstanceOf("unittest.mock.NonCallableMock"),
    TypeMatchers.isObjectInstanceOf("mock.mock.NonCallableMock"));

  private static final TypeMatcher ASYNC_MOCK_INSTANCE = TypeMatchers.any(
    TypeMatchers.isObjectInstanceOf("unittest.mock.AsyncMock"),
    TypeMatchers.isObjectInstanceOf("mock.mock.AsyncMock"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, MisspelledMockAssertionCheck::checkCall);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Expression callee = Expressions.removeParentheses(callExpression.callee());
    if (!(callee instanceof QualifiedExpression qualifiedExpression)) {
      return;
    }

    Name memberName = qualifiedExpression.name();
    String name = memberName.name();
    if (KNOWN_ASSERTION_METHODS.contains(name) || !name.startsWith("assert")) {
      return;
    }

    Expression qualifier = Expressions.removeParentheses(qualifiedExpression.qualifier());
    if (!isMockOrChildMock(qualifier, ctx)) {
      return;
    }
    // Exact declared members are not typos (e.g. a real assert_* helper on the mock type).
    if (TypeMatchers.hasMember(name).isTrueFor(qualifier, ctx)) {
      return;
    }

    String suggestion = findUniqueCloseMatch(name, isAsyncMockOrChildMock(qualifier, ctx));
    if (suggestion == null) {
      return;
    }

    var issue = ctx.addIssue(memberName, String.format(MESSAGE, suggestion));
    issue.addQuickFix(PythonQuickFix.newQuickFix(String.format(QUICK_FIX_MESSAGE, suggestion))
      .addTextEdit(TextEditUtils.replace(memberName, suggestion))
      .build());
  }

  private static boolean isMockOrChildMock(Expression expression, SubscriptionContext ctx) {
    return matchesAlongQualifierChain(expression, ctx, MOCK_INSTANCE);
  }

  private static boolean isAsyncMockOrChildMock(Expression expression, SubscriptionContext ctx) {
    return matchesAlongQualifierChain(expression, ctx, ASYNC_MOCK_INSTANCE);
  }

  private static boolean matchesAlongQualifierChain(Expression expression, SubscriptionContext ctx, TypeMatcher matcher) {
    Expression current = Expressions.removeParentheses(expression);
    while (true) {
      if (matcher.isTrueFor(current, ctx)) {
        return true;
      }
      if (!(current instanceof QualifiedExpression qualifiedExpression)) {
        return false;
      }
      current = Expressions.removeParentheses(qualifiedExpression.qualifier());
    }
  }

  @CheckForNull
  private static String findUniqueCloseMatch(String name, boolean asyncMock) {
    String bestMatch = null;
    int bestDistance = MAX_EDIT_DISTANCE + 1;
    boolean tied = false;
    for (String known : KNOWN_ASSERTION_METHODS) {
      if (isAwaitAssertion(known) && !asyncMock) {
        continue;
      }
      int distance = LEVENSHTEIN.apply(name, known);
      if (distance > 0) {
        if (distance < bestDistance) {
          bestDistance = distance;
          bestMatch = known;
          tied = false;
        } else if (distance == bestDistance) {
          tied = true;
        }
      }
    }
    return tied ? null : bestMatch;
  }

  private static boolean isAwaitAssertion(String methodName) {
    return methodName.contains("await");
  }
}
