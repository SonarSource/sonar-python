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

import java.text.ParsePosition;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
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

    if (calleeSymbol != null && calleeSymbol.fullyQualifiedName() != null && FQN_TO_CHECK.contains(calleeSymbol.fullyQualifiedName())) {
      extractPatternFromCallExpr(callExpression).ifPresent(pattern -> {
        checkForEllipsisInParenthesis(ctx, pattern);
        checkForUnbalancedParenthesis(ctx, pattern);
        checkForUnusedParameter(ctx, callExpression.arguments(), pattern);
      });
    }
  }

  private void checkForUnusedParameter(SubscriptionContext ctx, List<Argument> arguments, EinopsPattern pattern) {

    var argsToCheck = arguments.stream()
      .map(TreeUtils.toInstanceOfMapper(RegularArgument.class))
      .filter(Objects::nonNull)
      .filter(arg -> arg.expression().is(Tree.Kind.NUMERIC_LITERAL))
      .filter(arg -> arg.keywordArgument() != null)
      .map(arg -> arg.keywordArgument().name())
      .filter(argName -> !pattern.lhsIdentifiers().contains(argName))
      .filter(argName -> !pattern.rhsIdentifiers().contains(argName))
      .toList();

    if (!argsToCheck.isEmpty()) {
      var isPlural = argsToCheck.size() > 1;
      var missingParameters = argsToCheck.stream().collect(Collectors.joining(", "));
      var missingParametersMessage = String.format("the parameter%s %s do%s not appear in the pattern", isPlural ? "s" : "", missingParameters, isPlural ? "" : "es");
      ctx.addIssue(pattern.originalPattern(), String.format(MESSAGE_TEMPLATE, missingParametersMessage));
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

  private record EinopsPattern(StringLiteral originalPattern, String lhs, String rhs, Set<String> lhsIdentifiers, Set<String> rhsIdentifiers) {
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
      var lhs = split[0].trim();
      var rhs = split[1].trim();
      var lhsIdentifiers = extractIdentifiers(lhs);
      var rhsIdentifiers = extractIdentifiers(rhs);
      return Optional.of(new EinopsPattern(pattern, lhs, rhs, lhsIdentifiers, rhsIdentifiers));
    }
    return Optional.empty();
  }

  private static Set<String> extractIdentifiers(String pattern) {
    Set<String> identifiers = new LinkedHashSet<>();
    var currentIdentifier = new StringBuilder();
    for (int i = 0; i < pattern.length(); i++) {
      char c = pattern.charAt(i);
      if (c == ' ' || c == '(' || c == ')' && !currentIdentifier.isEmpty()) {
        identifiers.add(currentIdentifier.toString());
        currentIdentifier.setLength(0);
      }
      // \u2026 is the unicode character for Ellipsis
      if (Character.isLetterOrDigit(c) || c == '_' || c == '\u2026') {
        currentIdentifier.append(c);
      }
    }
    return identifiers;
  }
}
