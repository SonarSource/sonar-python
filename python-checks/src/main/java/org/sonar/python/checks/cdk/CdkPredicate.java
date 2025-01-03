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
package org.sonar.python.checks.cdk;

import java.util.Arrays;
import java.util.Collection;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

public class CdkPredicate {

  private CdkPredicate() {

  }

  /**
   * @return Predicate which tests if expression is boolean literal and is set to `false`
   */
  public static Predicate<Expression> isFalse() {
    return expression -> Optional.ofNullable(expression.firstToken()).map(Token::value).filter("False"::equals).isPresent();
  }

  /**
   * @return Predicate which tests if expression is boolean literal and is set to `true`
   */
  public static Predicate<Expression> isTrue() {
    return expression -> Optional.ofNullable(expression.firstToken()).map(Token::value).filter("True"::equals).isPresent();
  }

  /**
   * @return Predicate which tests if expression is `none`
   */
  public static Predicate<Expression> isNone() {
    return expression -> expression.is(Tree.Kind.NONE);
  }

  /**
   * @return Predicate which tests if expression is a fully qualified name (FQN) and is equal the expected FQN
   */
  public static Predicate<Expression> isFqn(String fqnValue) {
    return expression ->  TreeUtils.fullyQualifiedNameFromExpression(expression)
      .filter(fqnValue::equals)
      .isPresent();
  }

  /**
   * @return Predicate which tests if expression is a fully qualified name (FQN) and part of the FQN list
   */
  public static Predicate<Expression> isFqnOf(Collection<String> fqnValues) {
    return expression ->  TreeUtils.fullyQualifiedNameFromExpression(expression)
      .filter(fqnValues::contains)
      .isPresent();
  }

  /**
   * @return Predicate which tests if expression is a string and is equal the expected value
   */
  public static Predicate<Expression> isString(String expectedValue) {
    return expression -> CdkUtils.getString(expression).filter(expectedValue::equals).isPresent();
  }

  public static Predicate<Expression> isWildcard() {
    return isString("*");
  }

  /**
   * @return Predicate which tests if expression is a string and is equal to any of the expected values
   */
  public static Predicate<Expression> isString(Set<String> expectedValues) {
    return expression -> CdkUtils.getString(expression).filter(expectedValues::contains).isPresent();
  }

  /**
   * @return Predicate which tests if expression is a string matches the pattern
   */
  public static Predicate<Expression> matches(Pattern pattern) {
    return expression -> CdkUtils.getString(expression).filter(string -> pattern.matcher(string).find()).isPresent();
  }

  /**
   * @return Predicate which tests if expression is a string literal
   */
  public static Predicate<Expression> isStringLiteral() {
    return expression -> expression.is(Tree.Kind.STRING_LITERAL);
  }

  /**
   * @return Predicate which tests if expression is a number literal
   */
  public static Predicate<Expression> isNumericLiteral() {
    return expression -> expression.is(Tree.Kind.NUMERIC_LITERAL);
  }

  /**
   * @return Predicate which tests if expression is a list literal
   */
  public static Predicate<Expression> isListLiteral() {
    return expression -> expression.is(Tree.Kind.LIST_LITERAL);
  }

  public static Predicate<Expression> isSubscriptionExpression() {
    return expression -> expression.is(Tree.Kind.SUBSCRIPTION);
  }

  /**
   * @return Predicate which tests if expression is a string and starts with the expected value
   */
  public static Predicate<Expression> startsWith(String expected) {
    return expression -> CdkUtils.getString(expression).filter(str -> str.toLowerCase(Locale.ROOT).startsWith(expected)).isPresent();
  }

  public static Predicate<Expression> isCallExpression() {
    return expression -> expression.is(Tree.Kind.CALL_EXPR);
  }

  // Predicate on a Expression that is expected to be a CallExpression with a specific argument (name/pos) on which predicates are applicable
  @SafeVarargs
  public static Predicate<Expression> hasArgument(String name, int pos, Predicate<Expression>... predicates) {
    return expression -> {
      if (!expression.is(Tree.Kind.CALL_EXPR)) {
        return false;
      }

      return getArgument(null, (CallExpression) expression, name, pos)
        .filter(flow -> flow.hasExpression(Arrays.stream(predicates).reduce(x -> true, Predicate::and))).isPresent();
    };
  }

  @SafeVarargs
  public static Predicate<Expression> hasArgument(String name, Predicate<Expression>... predicates) {
    return hasArgument(name, -1, predicates);
  }

  public static Predicate<Expression> hasIntervalArguments(String argNameMin, int argPosMin, String argNameMax, int argPosMax, Collection<Long> values) {
    return expression -> {
      if (!expression.is(Tree.Kind.CALL_EXPR)) {
        return false;
      }

      Optional<Long> minVal = getArgumentAsLong((CallExpression) expression, argNameMin, argPosMin);
      Optional<Long> maxVal = getArgumentAsLong((CallExpression) expression, argNameMax, argPosMax);

      if (minVal.isEmpty() || maxVal.isEmpty()) {
        return false;
      }

      return anyValueInInterval(values, minVal.get(), maxVal.get());
    };
  }

  public static Predicate<Expression> hasIntervalArguments(String argNameMin, String argNameMax, Collection<Long> values) {
    return hasIntervalArguments(argNameMin, -1, argNameMax, -1, values);
  }

  private static Optional<Long> getArgumentAsLong(CallExpression callExpression, String argName, int argPos) {
    return getArgument(null, callExpression, argName, argPos)
      .flatMap(flow -> flow.getExpression(isNumericLiteral()))
      .map(NumericLiteral.class::cast)
      .map(NumericLiteral::valueAsLong);
  }

  private static boolean anyValueInInterval(Collection<Long> values, long min, long max) {
    for (long val : values) {
      if (min <= val && val <= max) {
        return true;
      }
    }
    return false;
  }

  public static Predicate<Expression> isNumeric(Set<Long> vals) {
    return expression -> expression.is(Tree.Kind.NUMERIC_LITERAL) && vals.contains(((NumericLiteral) expression).valueAsLong());
  }

}
