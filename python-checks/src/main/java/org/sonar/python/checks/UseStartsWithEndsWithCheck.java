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

import java.util.Map;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.SliceItem;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.NumericLiteralImpl;

import static org.sonar.plugins.python.api.tree.Tree.Kind.COMPARISON;
import static org.sonar.plugins.python.api.tree.Tree.Kind.SLICE_ITEM;

@Rule(key = "S6659")
public class UseStartsWithEndsWithCheck extends PythonSubscriptionCheck {
  private static final TypeMatcher STR_MATCHER = TypeMatchers.isObjectInstanceOf("builtins.str");
  private static final TypeMatcher NONE_MATCHER = TypeMatchers.isObjectOfType("NoneType");

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

    // Exactly one operand must be a slice expression.
    // The other one must be the string we compare it to.
    // If both sides are slices (e.g. a[:4] == b[:4]), there is no clean startswith/endswith replacement.
    var lhs = comparison.leftOperand();
    var rhs = comparison.rightOperand();
    final SliceExpression sliceExpression;
    final Expression stringExpression;
    if (lhs.is(Tree.Kind.SLICE_EXPR) && !rhs.is(Tree.Kind.SLICE_EXPR)) {
      sliceExpression = (SliceExpression) lhs;
      stringExpression = rhs;
    } else if (rhs.is(Tree.Kind.SLICE_EXPR) && !lhs.is(Tree.Kind.SLICE_EXPR)) {
      sliceExpression = (SliceExpression) rhs;
      stringExpression = lhs;
    } else {
      return;
    }

    // To avoid FPs, either the slice expression must slice a string, or the object we compare it to must clearly be a string.
    if (!STR_MATCHER.isTrueFor(stringExpression, ctx) &&
      !STR_MATCHER.isTrueFor(sliceExpression.object(), ctx)) {
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

    var sliceType = SliceType.fromSliceItem((SliceItem) sliceItem, ctx);

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

    private static SliceType fromSliceItem(SliceItem sliceItem, SubscriptionContext ctx) {
      var stride = sliceItem.stride();
      // If the stride is
      // not absent
      // and not None
      // and not the "1" literal
      // then we don't check the rule.
      if (stride != null &&
        !NONE_MATCHER.isTrueFor(stride, ctx) &&
        !isIntLiteralEqualTo(stride, 1)) {
        return SliceType.COMPLEX;
      }

      var lowerBound = sliceItem.lowerBound();
      var upperBound = sliceItem.upperBound();

      // Case [x:None:...]
      if (!isEmptyBound(lowerBound, ctx) &&
        isEmptyBound(upperBound, ctx)) {

        return SliceType.SUFFIX;
      }

      // Case [None:x:...]
      if (!isEmptyBound(upperBound, ctx) &&
        isEmptyBound(lowerBound, ctx)) {
        return SliceType.PREFIX;
      }

      return SliceType.COMPLEX;
    }

    private static boolean isEmptyBound(@CheckForNull Expression bound, SubscriptionContext ctx) {
      return bound == null || NONE_MATCHER.isTrueFor(bound, ctx);
    }
  }

  private static boolean isIntLiteralEqualTo(Expression expression, long expected) {
    try {
      return expression instanceof NumericLiteralImpl numericLiteral
        && numericLiteral.numericKind() == NumericLiteralImpl.NumericKind.INT
        && numericLiteral.valueAsLong() == expected;
    } catch (NumberFormatException nfe) {
      return false;
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
