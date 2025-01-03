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

import java.util.Map;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.SliceItem;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.BuiltinTypes;

import static org.sonar.plugins.python.api.tree.Tree.Kind.COMPARISON;
import static org.sonar.plugins.python.api.tree.Tree.Kind.SLICE_ITEM;

@Rule(key = "S6659")
public class UseStartsWithEndsWithCheck extends PythonSubscriptionCheck {
  private static final String USE_STARTSWITH_MESSAGE = "Use `startswith` here.";
  private static final String USE_NOT_STARTSWITH_MESSAGE = "Use `not` and `startswith` here.";
  private static final String USE_ENDSWITH_MESSAGE = "Use `endswith` here.";
  private static final String USE_NOT_ENDSWITH_MESSAGE = "Use `not` and `endswith` here.";
  private static final Map<SliceType, Map<OperatorType, String>> MESSAGES = Map.of(
    SliceType.PREFIX, Map.of(
      OperatorType.EQUALS, USE_STARTSWITH_MESSAGE,
      OperatorType.NOT_EQUALS, USE_NOT_STARTSWITH_MESSAGE),
    SliceType.SUFFIX, Map.of(
      OperatorType.EQUALS, USE_ENDSWITH_MESSAGE,
      OperatorType.NOT_EQUALS, USE_NOT_ENDSWITH_MESSAGE));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(COMPARISON, ctx -> checkComparison(ctx, ((BinaryExpression) ctx.syntaxNode())));
  }

  private static void checkComparison(SubscriptionContext ctx, BinaryExpression comparison) {
    var operatorType = OperatorType.fromString(comparison.operator().value());
    if (operatorType == OperatorType.OTHER) {
      return;
    }

    // Either the left or the right operand must be a slice expression.
    // The other one must be the string we compare it to:
    var lhs = comparison.leftOperand();
    var rhs = comparison.rightOperand();
    final SliceExpression sliceExpression;
    final Expression stringExpression;
    if (lhs.is(Tree.Kind.SLICE_EXPR)) {
      sliceExpression = (SliceExpression) lhs;
      stringExpression = rhs;
    } else if (rhs.is(Tree.Kind.SLICE_EXPR)) {
      sliceExpression = (SliceExpression) rhs;
      stringExpression = lhs;
    } else {
      return;
    }

    // To avoid FPs, either the slice expression must slice a string, or the object we compare it to must clearly be a string.
    if (!stringExpression.type().mustBeOrExtend(BuiltinTypes.STR) &&
      !sliceExpression.object().type().mustBeOrExtend(BuiltinTypes.STR)) {
      return;
    }

    var slices = sliceExpression.sliceList().slices();
    if (slices.size() != 1) {
      return;
    }

    var sliceItem = slices.get(0);
    if (!sliceItem.is(SLICE_ITEM)) {
      return;
    }

    var sliceType = SliceType.fromSliceItem((SliceItem) sliceItem);

    var message = selectMessage(sliceType, operatorType);
    if (message == null) {
      return;
    }

    ctx.addIssue(comparison, message);
  }

  @CheckForNull
  private static String selectMessage(SliceType sliceType, OperatorType operatorType) {
    var operatorMap = MESSAGES.get(sliceType);
    if (operatorMap == null) {
      return null;
    }

    return operatorMap.get(operatorType);
  }

  private enum SliceType {
    PREFIX,
    SUFFIX,
    COMPLEX;

    private static SliceType fromSliceItem(SliceItem sliceItem) {
      var stride = sliceItem.stride();
      // If the stride is
      // not absent
      // and not None
      // and not the "1" literal
      // then we don't check the rule.
      if (stride != null &&
        !stride.type().mustBeOrExtend(BuiltinTypes.NONE_TYPE) &&
        !(stride.is(Tree.Kind.NUMERIC_LITERAL) &&
          stride.type().mustBeOrExtend(BuiltinTypes.INT) &&
          ((NumericLiteral) stride).valueAsLong() == 1)) {
        return SliceType.COMPLEX;
      }

      var lowerBound = sliceItem.lowerBound();
      var upperBound = sliceItem.upperBound();

      // Case [x:None:...]
      if (!isEmptyBound(lowerBound) &&
        isEmptyBound(upperBound)) {

        return SliceType.SUFFIX;
      }

      // Case [None:x:...]
      if (!isEmptyBound(upperBound) &&
        isEmptyBound(lowerBound)) {
        return SliceType.PREFIX;
      }

      return SliceType.COMPLEX;
    }

    private static boolean isEmptyBound(@CheckForNull Expression bound) {
      return bound == null || bound.type().mustBeOrExtend(BuiltinTypes.NONE_TYPE);
    }
  }

  private enum OperatorType {
    EQUALS,
    NOT_EQUALS,
    OTHER;

    private static OperatorType fromString(String operator) {
      if ("==".equals(operator)) {
        return EQUALS;
      }

      if ("!=".equals(operator)) {
        return NOT_EQUALS;
      }

      return OTHER;
    }
  }
}
