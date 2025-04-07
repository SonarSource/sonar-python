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
package org.sonar.python.checks.hotspots;

import java.util.function.BiConsumer;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

public class CommonValidationUtils {

  private CommonValidationUtils() {
    // Prevent instantiation
  }

  static boolean isLessThan(Expression expression, int number) {
    try {
      if (expression.is(Tree.Kind.NAME)) {
        return Expressions.singleAssignedNonNameValue(((Name) expression)).map(value -> isLessThan(value, number)).orElse(false);
      }
      return expression.is(Tree.Kind.NUMERIC_LITERAL) && ((NumericLiteral) expression).valueAsLong() < number;
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  static boolean isLessThanExponent(Expression expression, int exponent) {
    if (expression.is(Tree.Kind.SHIFT_EXPR)) {
      var shiftExpression = (BinaryExpression) expression;
      return shiftExpression.leftOperand().is(Tree.Kind.NUMERIC_LITERAL) && (((NumericLiteral) shiftExpression.leftOperand()).valueAsLong() == 1)
        && isLessThan(shiftExpression.rightOperand(), exponent);
    }
    if (expression.is(Tree.Kind.NAME)) {
      return Expressions.singleAssignedNonNameValue(((Name) expression))
        .map(v -> isLessThanExponent(v, exponent))
        .orElse(false);
    }
    return false;
  }

  interface CallValidator {
    void validate(SubscriptionContext ctx, CallExpression callExpression);
  }

  record ArgumentValidator(
    int position,
    String keywordName,
    BiConsumer<SubscriptionContext, RegularArgument> consumer
  ) implements CallValidator {

    @Override
    public void validate(SubscriptionContext ctx, CallExpression callExpression) {
      TreeUtils.nthArgumentOrKeywordOptional(position, keywordName, callExpression.arguments())
        .ifPresent(argument -> consumer.accept(ctx, argument));
    }
  }
}
