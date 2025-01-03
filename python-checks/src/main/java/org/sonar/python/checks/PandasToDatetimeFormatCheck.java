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

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6894")
public class PandasToDatetimeFormatCheck extends PythonSubscriptionCheck {

  static final String PANDAS_TO_DATETIME_FQN = "pandas.core.tools.datetimes.to_datetime";
  static final String MESSAGE = "Remove this `%s=%s` parameter or make sure the provided date(s) can be parsed accordingly.";
  static final String SECONDARY_MESSAGE = "Invalid date.";
  static final String DAYFIRST = "dayfirst";
  static final String YEARFIRST = "yearfirst";

  static final Map<String, ParseResult> DATE_FORMATS = new HashMap<>();

  static {
    // Various separators can actually be provided to pandas.to_datetime and will be normalized to "-"
    // The absence of separator is treated separately as a special case to avoid loss of information (leading to possible FNs)
    DATE_FORMATS.put("yyyy-MM-dd", new ParseResult(true, false, false, true));
    DATE_FORMATS.put("yyyyMMdd", new ParseResult(true, false, false, true));
    DATE_FORMATS.put("yy-MM-dd", new ParseResult(true, false, false, true));
    DATE_FORMATS.put("yyMMdd", new ParseResult(true, false, false, true));
    DATE_FORMATS.put("yyyy-dd-MM", new ParseResult(true, false, true, false));
    DATE_FORMATS.put("yyyyddMM", new ParseResult(true, false, true, false));
    DATE_FORMATS.put("yy-dd-MM", new ParseResult(true, false, true, false));
    DATE_FORMATS.put("yyddMM", new ParseResult(true, false, true, false));
    DATE_FORMATS.put("dd-MM-yyyy", new ParseResult(false, true, true, false));
    DATE_FORMATS.put("ddMMyyyy", new ParseResult(false, true, true, false));
    DATE_FORMATS.put("dd-MM-yy", new ParseResult(false, true, true, false));
    DATE_FORMATS.put("ddMMyy", new ParseResult(false, true, true, false));
    DATE_FORMATS.put("MM-dd-yyyy", new ParseResult(false, true, false, true));
    DATE_FORMATS.put("MMddyyyy", new ParseResult(false, true, false, true));
    DATE_FORMATS.put("MM-dd-yy", new ParseResult(false, true, false, true));
    DATE_FORMATS.put("MMddyy", new ParseResult(false, true, false, true));
  }


  static final List<String> TIME_FORMATS = List.of(
    "'T'HH:mm:ss",
    "'T'HH:mm",
    "-HH:mm:ss",
    "-HH:mm",
    "HH:mm:ss",
    "HH:mm"
  );
  final Map<String, DateTimeFormatter> formatters = new HashMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Symbol symbol = callExpression.calleeSymbol();
    if (symbol == null || !PANDAS_TO_DATETIME_FQN.equals(symbol.fullyQualifiedName())) {
      return;
    }
    RegularArgument argument = TreeUtils.nthArgumentOrKeyword(0, "arg", callExpression.arguments());
    if (argument == null) {
      return;
    }
    Expression argumentExpression = argument.expression();
    List<ExpressionAndStringValue> expressionAndStringValues = getExpressionsAndStringValues(argumentExpression);

    RegularArgument dayfirstArgument = TreeUtils.nthArgumentOrKeyword(2, DAYFIRST, callExpression.arguments());
    RegularArgument yearfirstArgument = TreeUtils.nthArgumentOrKeyword(3, YEARFIRST, callExpression.arguments());

    this.checkArguments(ctx, dayfirstArgument, yearfirstArgument, expressionAndStringValues, argumentExpression);
  }

  private void checkArguments(SubscriptionContext ctx, @Nullable RegularArgument dayfirstArgument, @Nullable RegularArgument yearfirstArgument,
    List<ExpressionAndStringValue> expressionAndStringValues, Expression argumentExpression) {

    boolean isDayFirstTrue = getArgumentConstraint(dayfirstArgument, Expressions::isTruthy);
    boolean isDayFirstFalse = getArgumentConstraint(dayfirstArgument, Expressions::isFalsy);
    boolean isYearFirstTrue = getArgumentConstraint(yearfirstArgument, Expressions::isTruthy);
    boolean isYearFirstFalse = getArgumentConstraint(yearfirstArgument, Expressions::isFalsy);

    // False flags on expected parse results represent unmet conditions. Flags are true by default if no constraint is specified
    ParseResult expectedParseResult = new ParseResult(!isYearFirstTrue, !isYearFirstFalse, !isDayFirstTrue, !isDayFirstFalse);

    if (expectedParseResult.areAllConditionsMet()) {
      return;
    }

    for (ExpressionAndStringValue expressionAndStringValue : expressionAndStringValues) {
      ParseResult parseResult = this.parseResult(expressionAndStringValue.normalizedStringValue, expectedParseResult);

      if (dayfirstArgument != null && !parseResult.isCompatibleDayFirstTrue) {
        reportIssue(ctx, dayfirstArgument, argumentExpression, expressionAndStringValue.originalExpression, String.format(MESSAGE, DAYFIRST, "True"));
        return;
      }
      if (dayfirstArgument != null && !parseResult.isCompatibleDayFirstFalse) {
        reportIssue(ctx, dayfirstArgument, argumentExpression, expressionAndStringValue.originalExpression, String.format(MESSAGE, DAYFIRST, "False"));
        return;
      }

      if (yearfirstArgument != null && !parseResult.isCompatibleYearFirstTrue) {
        reportIssue(ctx, yearfirstArgument, argumentExpression, expressionAndStringValue.originalExpression, String.format(MESSAGE, YEARFIRST, "True"));
        return;
      }
      if (yearfirstArgument != null && !parseResult.isCompatibleYearFirstFalse) {
        reportIssue(ctx, yearfirstArgument, argumentExpression, expressionAndStringValue.originalExpression, String.format(MESSAGE, YEARFIRST, "False"));
        return;
      }
    }
  }

  private static void reportIssue(SubscriptionContext ctx, RegularArgument dayfirstArgument, Expression argumentExpression, Expression originalExpression, String message) {
    PreciseIssue preciseIssue = ctx.addIssue(dayfirstArgument, message);
    if (argumentExpression != originalExpression) {
      preciseIssue.secondary(argumentExpression, "This contains invalid date(s).");
      preciseIssue.secondary(originalExpression, SECONDARY_MESSAGE);
    } else {
      preciseIssue.secondary(argumentExpression, SECONDARY_MESSAGE);
    }
  }

  private static boolean getArgumentConstraint(@Nullable RegularArgument dayfirstArgument, Predicate<Expression> predicate) {
    return Optional.ofNullable(dayfirstArgument)
      .map(RegularArgument::expression).map(e -> {
          Expression returnValue = e;
          if (e.is(Tree.Kind.NAME) && isNotABooleanValue(e)) {
            returnValue = Expressions.singleAssignedValue(((Name) e));
          }
          return returnValue;
        }
      ).map(predicate::test).orElse(false);
  }

  private static boolean isNotABooleanValue(Expression e) {
    return !Expressions.isTruthy(e) && !Expressions.isFalsy(e);
  }

  private static List<ExpressionAndStringValue> getExpressionsAndStringValues(Expression expression) {
    if (expression.is(Tree.Kind.STRING_LITERAL)) {
      String normalizedStringValue = normalizeDateString(((StringLiteral) expression).trimmedQuotesValue());
      return List.of(new ExpressionAndStringValue(expression, normalizedStringValue));
    }
    if (expression.is(Tree.Kind.NAME)) {
      Optional<Expression> assignedValue = Expressions.singleAssignedNonNameValue((Name) expression);
      return assignedValue.map(PandasToDatetimeFormatCheck::getExpressionsAndStringValues).orElse(Collections.emptyList());
    }
    if (expression.is(Tree.Kind.LIST_LITERAL)) {
      return ((ListLiteral) expression).elements().expressions().stream()
        .map(PandasToDatetimeFormatCheck::getExpressionsAndStringValues).flatMap(List::stream).toList();
    }
    return Collections.emptyList();
  }

  private static String normalizeDateString(String dateString) {
    return dateString.trim().replaceAll("[\\./;\\s_]", "-");
  }

  private ParseResult parseResult(String normalizedDateString, ParseResult expected) {
    boolean parsedOnce = false;
    for (Map.Entry<String, ParseResult> entry : DATE_FORMATS.entrySet()) {
      if (expected.areAllConditionsMet()) {
        return expected;
      }
      String dateFormat = entry.getKey();
      ParseResult parseResult = entry.getValue();
      DateTimeFormatter dateTimeFormatter = formatters.computeIfAbsent(dateFormat, DateTimeFormatter::ofPattern);
      try {
        LocalDate.parse(normalizedDateString, dateTimeFormatter);
        parsedOnce = true;
        expected = expected.updateExpectedParseResult(parseResult);
      } catch (DateTimeParseException e) {
        // Incorrect format, continue
      }
      for (String timeFormat : TIME_FORMATS) {
        try {
          String dateTimeFormat = dateFormat + timeFormat;
          dateTimeFormatter = formatters.computeIfAbsent(dateTimeFormat, DateTimeFormatter::ofPattern);
          LocalDateTime.parse(normalizedDateString, dateTimeFormatter);
          parsedOnce = true;
          expected = expected.updateExpectedParseResult(parseResult);
        } catch (DateTimeParseException e) {
          // Incorrect format, continue
        }
      }
    }
    if (parsedOnce) {
      return expected;
    }
    // Can't be parsed: no issue raised
    return new ParseResult(true, true, true, true);
  }

  static class ExpressionAndStringValue {
    Expression originalExpression;
    String normalizedStringValue;

    public ExpressionAndStringValue(Expression originalExpression, String normalizedStringValue) {
      this.originalExpression = originalExpression;
      this.normalizedStringValue = normalizedStringValue;
    }
  }

  static final class ParseResult {
    final boolean isCompatibleYearFirstTrue;
    final boolean isCompatibleYearFirstFalse;
    final boolean isCompatibleDayFirstTrue;
    final boolean isCompatibleDayFirstFalse;

    ParseResult(boolean isCompatibleYearFirstTrue, boolean isCompatibleYearFirstFalse, boolean isCompatibleDayFirstTrue, boolean isCompatibleDayFirstFalse) {
      this.isCompatibleYearFirstTrue = isCompatibleYearFirstTrue;
      this.isCompatibleYearFirstFalse = isCompatibleYearFirstFalse;
      this.isCompatibleDayFirstTrue = isCompatibleDayFirstTrue;
      this.isCompatibleDayFirstFalse = isCompatibleDayFirstFalse;
    }

    public boolean areAllConditionsMet() {
      return isCompatibleYearFirstTrue && isCompatibleYearFirstFalse && isCompatibleDayFirstTrue && isCompatibleDayFirstFalse;
    }

    ParseResult updateExpectedParseResult(ParseResult parseResult) {
      return new ParseResult(
        isCompatibleYearFirstTrue || parseResult.isCompatibleYearFirstTrue,
        isCompatibleYearFirstFalse || parseResult.isCompatibleYearFirstFalse,
        isCompatibleDayFirstTrue || parseResult.isCompatibleDayFirstTrue,
        isCompatibleDayFirstFalse || parseResult.isCompatibleDayFirstFalse
      );
    }
  }

}
