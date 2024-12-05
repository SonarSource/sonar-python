/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.checks.regex;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.regex.PythonRegexIssueLocation;
import org.sonar.python.regex.RegexContext;
import org.sonar.python.tree.TreeUtils;
import org.sonarsource.analyzer.commons.regex.RegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.FlagSet;
import org.sonarsource.analyzer.commons.regex.ast.RegexSyntaxElement;

public abstract class AbstractRegexCheck extends PythonSubscriptionCheck {

  private static final Map<String, Integer> REGEX_FUNCTIONS_TO_FLAG_PARAM = new HashMap<>();

  static {
    REGEX_FUNCTIONS_TO_FLAG_PARAM.put("re.sub", 4);
    REGEX_FUNCTIONS_TO_FLAG_PARAM.put("re.subn", 4);
    REGEX_FUNCTIONS_TO_FLAG_PARAM.put("re.compile", 1);
    REGEX_FUNCTIONS_TO_FLAG_PARAM.put("re.search", 2);
    REGEX_FUNCTIONS_TO_FLAG_PARAM.put("re.match", 2);
    REGEX_FUNCTIONS_TO_FLAG_PARAM.put("re.fullmatch", 2);
    REGEX_FUNCTIONS_TO_FLAG_PARAM.put("re.split", 3);
    REGEX_FUNCTIONS_TO_FLAG_PARAM.put("re.findall", 2);
    REGEX_FUNCTIONS_TO_FLAG_PARAM.put("re.finditer", 2);
  }

  protected RegexContext regexContext;

  // We want to report only one issue per element for one rule.
  protected final Set<RegexSyntaxElement> reportedRegexTrees = new HashSet<>();

  /**
   * Should return a map whose keys are the functions the check is interested in, and the values are the position of the flags parameter.
   * Set the position of the flags parameter to {@code null} if there is none.
   */
  protected Map<String, Integer> lookedUpFunctions() {
    return REGEX_FUNCTIONS_TO_FLAG_PARAM;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCall);
  }

  public abstract void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall);

  private void checkCall(SubscriptionContext ctx) {
    regexContext = (RegexContext) ctx;
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol == null || calleeSymbol.fullyQualifiedName() == null) {
      return;
    }
    String functionFqn = calleeSymbol.fullyQualifiedName();
    if (functionFqn != null && lookedUpFunctions().containsKey(functionFqn)) {
      FlagSet flagSet = getFlagSet(callExpression, functionFqn);

      patternArgStringLiteral(callExpression)
        .flatMap(l -> regexForStringLiteral(l, flagSet))
        .ifPresent(parseResult -> checkRegex(parseResult, callExpression));
    }
  }

  private Optional<RegexParseResult> regexForStringLiteral(StringLiteral literal, FlagSet flagSet) {
    if (shouldHandleStringLiteral(literal)) {
      return Optional.of(regexContext.regexForStringElement(literal.stringElements().get(0), flagSet));
    }
    return Optional.empty();
  }

  /**
   * We do ignore strings in the following cases:
   *  - It is a concatenation of multiple elements.
   *  - It is an f-string containing expressions. We don't have a good mechanism to evaluate these expressions currently.
   *  - The string is not raw and contains a \N{UNICODE NAME} escape sequence. In Java 8 we cannot make use of Character.codePointOf in the character parser (SONARPY-922).
   */
  private static boolean shouldHandleStringLiteral(StringLiteral literal) {
    if (literal.stringElements().size() != 1) {
      // We do not handle concatenations for now
      return false;
    }
    StringElement stringElement = literal.stringElements().get(0);
    return stringElement.formattedExpressions().isEmpty() &&
      (stringElement.prefix().toLowerCase(Locale.ROOT).contains("r") || !stringElement.value().contains("\\N{"));
  }

  private static Optional<StringLiteral> patternArgStringLiteral(CallExpression regexFunctionCall) {
    RegularArgument patternArgument = TreeUtils.nthArgumentOrKeyword(0, "pattern", regexFunctionCall.arguments());
    if (patternArgument == null) {
      return Optional.empty();
    }
    Expression patternValueExpression = patternArgument.expression();
    if (patternValueExpression.is(Tree.Kind.NAME)) {
      patternValueExpression = Expressions.singleAssignedValue((Name) patternValueExpression);
    }

    if (patternValueExpression != null && patternValueExpression.is(Tree.Kind.STRING_LITERAL)) {
      return Optional.of((StringLiteral) patternValueExpression);
    }
    return Optional.empty();
  }

  private FlagSet getFlagSet(CallExpression callExpression, String functionFqn) {
    HashSet<QualifiedExpression> flags = new HashSet<>();
    getFlagsArgValue(callExpression, lookedUpFunctions().get(functionFqn)).ifPresent(f -> flags.addAll(extractFlagExpressions(f)));
    FlagSet flagSet = new FlagSet();
    flags.stream()
      .map(AbstractRegexCheck::mapPythonFlag)
      .filter(Optional::isPresent)
      .map(Optional::get)
      .forEach(flagSet::add);

    // TODO: Don't do this when PYTHON_VERSION is 2
    // We used Pattern.LITERAL to represent re.ASCII. So we are checking if re.ASCII is set here.
    // For python3 matches are Unicode by default, and re.ASCII can be used to deactivate that.
    if (!flagSet.contains(Pattern.LITERAL)) {
      flagSet.add(Pattern.UNICODE_CHARACTER_CLASS);
      flagSet.add(Pattern.UNICODE_CASE);
    }
    flagSet.removeAll(new FlagSet(Pattern.LITERAL));

    return flagSet;
  }

  private static Optional<Expression> getFlagsArgValue(CallExpression regexFunctionCall, @Nullable Integer argPosition) {
    if (argPosition == null) {
      return Optional.empty();
    }
    RegularArgument patternArgument = TreeUtils.nthArgumentOrKeyword(argPosition, "flags", regexFunctionCall.arguments());
    return patternArgument != null ? Optional.of(patternArgument.expression()) : Optional.empty();
  }

  private static HashSet<QualifiedExpression> extractFlagExpressions(Tree flagsSubexpr) {
    if (flagsSubexpr.is(Tree.Kind.QUALIFIED_EXPR)) {
      return new HashSet<>(Collections.singletonList((QualifiedExpression) flagsSubexpr));
    } else if (flagsSubexpr.is(Tree.Kind.BITWISE_OR)) {
      // recurse into left and right branch
      BinaryExpression orExpr = (BinaryExpression) flagsSubexpr;
      HashSet<QualifiedExpression> flags = extractFlagExpressions(orExpr.leftOperand());
      flags.addAll(extractFlagExpressions(orExpr.rightOperand()));
      return flags;
    } else {
      // failed to interpret. Ignore leaf.
      return new HashSet<>();
    }
  }

  public static Optional<Integer> mapPythonFlag(QualifiedExpression ch) {
    Symbol symbol = ch.symbol();
    if (symbol == null) {
      return Optional.empty();
    }
    String symbolFqn = symbol.fullyQualifiedName();
    if (symbolFqn == null) {
      return Optional.empty();
    }

    Integer result = switch (symbolFqn) {
      case "re.IGNORECASE", "re.I" -> Pattern.CASE_INSENSITIVE;
      case "re.MULTILINE", "re.M" -> Pattern.MULTILINE;
      case "re.DOTALL", "re.S" -> Pattern.DOTALL;
      case "re.VERBOSE", "re.X" -> Pattern.COMMENTS;
      case "re.UNICODE", "re.U" -> Pattern.UNICODE_CHARACTER_CLASS;
      case "re.ASCII", "re.A" ->
        // We misuse Pattern.LITERAL to represent re.ASCII. It will be removed before being provided to the parser.
        Pattern.LITERAL;
      default -> null;
    };
    return Optional.ofNullable(result);
  }

  public PreciseIssue addIssue(RegexSyntaxElement regexTree, String message, @Nullable Integer cost, List<RegexIssueLocation> secondaries) {
    if (reportedRegexTrees.add(regexTree)) {
      PreciseIssue issue = regexContext.addIssue(regexTree, message);
      secondaries.stream().map(PythonRegexIssueLocation::preciseLocation).forEach(issue::secondary);
      // TODO: Add cost to the issue
      return issue;
    }
    return null;
  }
}
