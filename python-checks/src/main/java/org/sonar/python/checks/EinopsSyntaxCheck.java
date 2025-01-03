/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
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
  private static final String NESTED_PARENTHESIS_MESSAGE = "nested parenthesis are not allowed";
  private static final String LHS_ELLIPSIS_MESSAGE = "Ellipsis inside parenthesis on the left side is not allowed";
  private static final String UNBALANCED_PARENTHESIS_MESSAGE = "parenthesis are unbalanced";
  private static final Set<String> FQN_TO_CHECK = Set.of("einops.repeat", "einops.reduce", "einops.rearrange");
  private static final Pattern ellipsisPattern = Pattern.compile("\\((.*)\\)");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, EinopsSyntaxCheck::checkEinopsSyntax);
  }

  private static void checkEinopsSyntax(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol != null && calleeSymbol.fullyQualifiedName() != null && FQN_TO_CHECK.contains(calleeSymbol.fullyQualifiedName())) {
      extractPatternFromCallExpr(callExpression).ifPresent(stringLiteral -> {
        var maybePattern = toEinopsPattern(stringLiteral);
        if (maybePattern.isPresent()) {
          var pattern = maybePattern.get();
          checkForEllipsisInParenthesis(ctx, pattern);
          checkForUnbalancedParenthesis(ctx, pattern);
          checkForUnusedParameter(ctx, callExpression.arguments(), pattern);
        } else {
          ctx.addIssue(callExpression.callee(), "Provide a valid einops pattern.");
        }
      });
    }
  }

  private static void checkForUnusedParameter(SubscriptionContext ctx, List<Argument> arguments, EinopsPattern pattern) {
    List<String> argsToCheck = arguments.stream()
      .map(TreeUtils.toInstanceOfMapper(RegularArgument.class))
      .filter(Objects::nonNull)
      .filter(arg -> arg.expression().is(Tree.Kind.NUMERIC_LITERAL))
      .filter(arg -> arg.keywordArgument() != null)
      .map(arg -> arg.keywordArgument().name())
      .filter(argName -> !pattern.lhs.identifiers.contains(argName))
      .filter(argName -> !pattern.rhs.identifiers.contains(argName))
      .toList();

    if (!argsToCheck.isEmpty()) {
      var isPlural = argsToCheck.size() > 1;
      var missingParameters = argsToCheck.stream().collect(Collectors.joining(", "));
      var missingParametersMessage = String.format("the parameter%s %s do%s not appear in the pattern", isPlural ? "s" : "", missingParameters, isPlural ? "" : "es");
      ctx.addIssue(pattern.originalPattern(), String.format(MESSAGE_TEMPLATE, missingParametersMessage));
    }
  }

  private static void checkForUnbalancedParenthesis(SubscriptionContext ctx, EinopsPattern pattern) {
    pattern.lhs.state.errorMessage.or(() -> pattern.rhs.state.errorMessage)
      .ifPresent(message -> ctx.addIssue(pattern.originalPattern(), String.format(MESSAGE_TEMPLATE, message)));
  }

  private static void checkForEllipsisInParenthesis(SubscriptionContext ctx, EinopsPattern pattern) {
    Matcher m = ellipsisPattern.matcher(pattern.lhs.originalPattern);
    if (m.find() && (m.group().contains("...") || m.group().contains("…"))) {
      ctx.addIssue(pattern.originalPattern(), String.format(MESSAGE_TEMPLATE, LHS_ELLIPSIS_MESSAGE));
    }
  }

  private record EinopsPattern(StringLiteral originalPattern, EinopsSide lhs, EinopsSide rhs) {
  }

  private record EinopsSide(String originalPattern, Set<String> identifiers, ParenthesisState state) {
  }

  private record ParenthesisState(boolean hasOpenParenthesis, Optional<String> errorMessage) {
  }

  private static Optional<StringLiteral> extractPatternFromCallExpr(CallExpression callExpression) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(1, "pattern", callExpression.arguments()))
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class));
  }

  private static Optional<EinopsPattern> toEinopsPattern(StringLiteral pattern) {
    String[] split = pattern.trimmedQuotesValue().split("->");
    if (split.length == 2) {
      var lhsStr = split[0].trim();
      var rhsStr = split[1].trim();
      if (!lhsStr.isEmpty() && !rhsStr.isEmpty()) {
        var lhs = parseEinopsPattern(lhsStr);
        var rhs = parseEinopsPattern(rhsStr);
        return Optional.of(new EinopsPattern(pattern, lhs, rhs));
      }
    }
    return Optional.empty();
  }

  private static EinopsSide parseEinopsPattern(String pattern) {
    Set<String> identifiers = new LinkedHashSet<>();
    var currentIdentifier = new StringBuilder();
    ParenthesisState state = new ParenthesisState(false, Optional.empty());
    for (int i = 0; i < pattern.length(); i++) {
      char c = pattern.charAt(i);
      if (c == ' ' || c == '(' || c == ')') {
        if (!currentIdentifier.isEmpty()) {
          identifiers.add(currentIdentifier.toString());
          currentIdentifier.setLength(0);
        }
        state = checkParenthesisBalance(c, state);
      } else if (Character.isLetterOrDigit(c) || c == '_' || c == '…') {
        currentIdentifier.append(c);
      }
    }
    if (!currentIdentifier.isEmpty()) {
      identifiers.add(currentIdentifier.toString());
    }
    if (state.hasOpenParenthesis && state.errorMessage.isEmpty()) {
      state = new ParenthesisState(true, Optional.of(UNBALANCED_PARENTHESIS_MESSAGE));
    }
    return new EinopsSide(pattern, identifiers, state);
  }

  private static ParenthesisState checkParenthesisBalance(char c, ParenthesisState state) {
    Optional<String> errorMessage = state.errorMessage;
    if (' ' == c) {
      return state;
    }
    if ('(' == c && state.hasOpenParenthesis) {
      errorMessage = Optional.of(NESTED_PARENTHESIS_MESSAGE);
    }
    if (')' == c && !state.hasOpenParenthesis && errorMessage.isEmpty()) {
      errorMessage = Optional.of(UNBALANCED_PARENTHESIS_MESSAGE);
    }
    return new ParenthesisState('(' == c, errorMessage);
  }
}
