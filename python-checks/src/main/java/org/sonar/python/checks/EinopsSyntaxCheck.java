/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks;

import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6984")
public class EinopsSyntaxCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TEMPLATE = "Fix the syntax of this einops operation: %s.";
  private static final String UNBALANCED_PARENTHESIS_MESSAGE = "parenthesis are unbalanced"; 
  private static final Set<String> FQN_TO_CHECK = Set.of("einops.repeat", "einops.reduce", "einops.rearrange");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkEinopsSyntax);
  }

  private void checkEinopsSyntax(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    Symbol calleeSymbol = callExpression.calleeSymbol();

    if (calleeSymbol != null && FQN_TO_CHECK.contains(calleeSymbol.fullyQualifiedName())) {
      extractPatternFromCallExpr(callExpression).ifPresent(pattern -> {
        checkForEllipsisInParenthesis(ctx, pattern);
        checkForUnbalancedParenthesis(ctx, pattern);
      });
    }
  }


  private void checkForUnbalancedParenthesis(SubscriptionContext ctx, EinopsPattern pattern) {
    hasUnbalancedParenthesis(pattern.lhs())
      .or(() -> hasUnbalancedParenthesis(pattern.rhs()))
      .ifPresent(message -> ctx.addIssue(pattern.originalPattern(), String.format(MESSAGE_TEMPLATE, message)));

  }

  private static Optional<String> hasUnbalancedParenthesis(String pattern) {
    boolean isBalanced = true;
    for (int i = 0; i < pattern.length(); i++) {
      char c = pattern.charAt(i);
      if ('(' == c) {
        if (!isBalanced) {
          return Optional.of("nested parenthesis are not allowed");
        }
        isBalanced = false;
        continue;
      }
      if (')' == c) {
        if (isBalanced) {
          return Optional.of(UNBALANCED_PARENTHESIS_MESSAGE);
        }
        isBalanced = true;
      }
    }
    if (!isBalanced) {
      return Optional.of(UNBALANCED_PARENTHESIS_MESSAGE);
    }
    return Optional.empty();
  }

  private record EinopsPattern(StringLiteral originalPattern, String lhs, String rhs) {
  }

  private static Pattern ellipsisPattern = Pattern.compile("\\(.*\\.{3}.*\\)");

  private void checkForEllipsisInParenthesis(SubscriptionContext ctx, EinopsPattern pattern) {
    if (ellipsisPattern.matcher(pattern.lhs).find()) {
      ctx.addIssue(pattern.originalPattern(), String.format(MESSAGE_TEMPLATE, "Ellipsis inside parenthesis on the left side is not allowed"));
    }
  }

  private static Optional<EinopsPattern> extractPatternFromCallExpr(CallExpression callExpression) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(1, "pattern", callExpression.arguments()))
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .flatMap(EinopsSyntaxCheck::toEinopsPattern);
  }

  private static Optional<EinopsPattern> toEinopsPattern(StringLiteral pattern) {
    String[] split = pattern.trimmedQuotesValue().split("->");
    if (split.length == 2) {
      return Optional.of(new EinopsPattern(pattern, split[0].trim(), split[1].trim()));
    }
    return Optional.empty();
  }
}
