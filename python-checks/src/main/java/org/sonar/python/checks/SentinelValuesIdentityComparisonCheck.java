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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.types.v2.matchers.InternalTypeMatchers;

@Rule(key = "S9462")
public class SentinelValuesIdentityComparisonCheck extends PythonSubscriptionCheck {

  public static final String IS_QUICK_FIX_MESSAGE = "Replace with \"is\"";
  public static final String IS_NOT_QUICK_FIX_MESSAGE = "Replace with \"is not\"";
  private static final TypeMatcher TYPING_EXTENSIONS_SENTINEL = TypeMatchers.isObjectOfType("typing_extensions.sentinel");
  private static final TypeMatcher SENTINEL = TypeMatchers.isObjectOfType("sentinel");
  private static final TypeMatcher SENTINEL_VALUE = TypeMatchers.any(
    TypeMatchers.isObjectOfType("builtins.sentinel"),
    TYPING_EXTENSIONS_SENTINEL,
    SENTINEL,
    InternalTypeMatchers.isAnyTypeInUnionSatisfying(SENTINEL));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.COMPARISON, SentinelValuesIdentityComparisonCheck::checkComparison);
  }

  private static void checkComparison(SubscriptionContext ctx) {
    var comparison = (BinaryExpression) ctx.syntaxNode();
    String operator = comparison.operator().value();
    if (("==".equals(operator) || "!=".equals(operator)) &&
      (SENTINEL_VALUE.isTrueFor(comparison.leftOperand(), ctx) || SENTINEL_VALUE.isTrueFor(comparison.rightOperand(), ctx))) {
      String replacement = "==".equals(operator) ? "is" : "is not";
      String quickFixMessage = "==".equals(operator) ? IS_QUICK_FIX_MESSAGE : IS_NOT_QUICK_FIX_MESSAGE;
      var quickFix = PythonQuickFix.newQuickFix(quickFixMessage)
        .addTextEdit(TextEditUtils.replace(comparison.operator(), replacementWithSpaces(comparison, replacement)))
        .build();
      ctx.addIssue(comparison.operator(), String.format("Use \"%s\" instead of \"%s\" when comparing with a sentinel value.", replacement, operator))
        .addQuickFix(quickFix);
    }
  }

  private static String replacementWithSpaces(BinaryExpression comparison, String replacement) {
    var operator = comparison.operator();
    var left = comparison.leftOperand().lastToken();
    var right = comparison.rightOperand().firstToken();
    boolean hasSpaceBefore = left.line() != operator.line() || left.column() + left.value().length() < operator.column();
    boolean hasSpaceAfter = right.line() != operator.line() || operator.column() + operator.value().length() < right.column();
    return (hasSpaceBefore ? "" : " ") + replacement + (hasSpaceAfter ? "" : " ");
  }
}
