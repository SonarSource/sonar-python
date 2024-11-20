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
package org.sonar.python.checks;

import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.InExpression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1940")
public class BooleanCheckNotInvertedCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use the opposite operator (\"%s\") instead.";

  private static final Set<String> EQUALITY_COMPARATORS = Set.of("==", "!=");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.NOT, ctx -> checkNotExpression(ctx, (UnaryExpression) ctx.syntaxNode()));
  }

  private static void checkNotExpression(SubscriptionContext ctx, UnaryExpression original) {
    Expression negatedExpr = original.expression();
    while (negatedExpr.is(Kind.PARENTHESIZED)) {
      negatedExpr = ((ParenthesizedExpression) negatedExpr).expression();
    }
    if (negatedExpr.is(Kind.COMPARISON)) {
      BinaryExpression binaryExp = (BinaryExpression) negatedExpr;
      // Don't raise warning with "not a == b == c" because a == b != c is not equivalent
      if (!binaryExp.leftOperand().is(Kind.COMPARISON)) {
        if (isSetComparison(binaryExp) && !EQUALITY_COMPARATORS.contains(binaryExp.operator().value())) {
          return;
        }
        String oppositeOperator = oppositeOperator(binaryExp.operator());

        PreciseIssue issue = (ctx.addIssue(original, String.format(MESSAGE, oppositeOperator)));
        createQuickFix(issue, oppositeOperator, binaryExp, original);
      }
    } else if (negatedExpr.is(Kind.IN, Kind.IS)) {
      BinaryExpression isInExpr = (BinaryExpression) negatedExpr;
      String oppositeOperator = oppositeOperator(isInExpr.operator(), isInExpr);

      PreciseIssue issue = (ctx.addIssue(original, String.format(MESSAGE, oppositeOperator)));
      createQuickFix(issue, oppositeOperator, isInExpr, original);
    }
  }

  private static String oppositeOperator(Token operator) {
    return oppositeOperatorString(operator.value());
  }

  private static String oppositeOperator(Token operator, Expression expr) {
    String s = operator.value();
    if (expr.is(Kind.IS) && ((IsExpression) expr).notToken() != null) {
      s = s + " not";
    } else if (expr.is(Kind.IN) && ((InExpression) expr).notToken() != null) {
      s = "not " + s;
    }
    return oppositeOperatorString(s);
  }

  static String oppositeOperatorString(String stringOperator) {
    switch (stringOperator) {
      case ">":
        return "<=";
      case ">=":
        return "<";
      case "<":
        return ">=";
      case "<=":
        return ">";
      case "==":
        return "!=";
      case "!=":
        return "==";
      case "is":
        return "is not";
      case "is not":
        return "is";
      case "in":
        return "not in";
      case "not in":
        return "in";
      default:
        throw new IllegalArgumentException("Unknown comparison operator : " + stringOperator);
    }
  }

  private static void createQuickFix(PreciseIssue issue, String oppositeOperator, BinaryExpression toUse, UnaryExpression notAncestor) {
    PythonTextEdit replaceEdit = getReplaceEdit(toUse, oppositeOperator, notAncestor);

    PythonQuickFix quickFix = PythonQuickFix.newQuickFix(String.format("Use %s instead", oppositeOperator))
      .addTextEdit(replaceEdit)
      .build();
    issue.addQuickFix(quickFix);
  }

  private static PythonTextEdit getReplaceEdit(BinaryExpression toUse, String oppositeOperator, UnaryExpression notAncestor) {
    return TextEditUtils.replace(notAncestor, getNewExpression(toUse, oppositeOperator));
  }

  private static String getNewExpression(BinaryExpression toUse, String oppositeOperator) {
    return getText(toUse.leftOperand()) + " " + oppositeOperator + " " + getText(toUse.rightOperand());
  }

  private static String getText(Tree defaultValue) {
    var tokens = TreeUtils.tokens(defaultValue);

    var valueBuilder = new StringBuilder();
    for (int i = 0; i < tokens.size(); i++) {
      var token = tokens.get(i);
      if (i > 0) {
        var previous = tokens.get(i - 1);
        var linesBetween = token.line() - previous.line();
        var spacesBetween = linesBetween == 0 ? (token.column() - previous.column() - previous.value().length()) : token.column();

        valueBuilder.append("\n".repeat(linesBetween));
        valueBuilder.append(" ".repeat(spacesBetween));
      }
      valueBuilder.append(token.value());
    }
    return valueBuilder.toString();
  }

  private static boolean isSetComparison(BinaryExpression binaryExpression) {
    List<InferredType> inferredTypeSet = List.of(
      binaryExpression.leftOperand().type(),
      binaryExpression.rightOperand().type()
    );
    return inferredTypeSet.stream().anyMatch(inferredType -> inferredType.mustBeOrExtend(BuiltinTypes.SET));
  }
}
