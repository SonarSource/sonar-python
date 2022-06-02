/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.InExpression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.tree.InExpressionImpl;
import org.sonar.python.tree.IsExpressionImpl;

import static org.sonar.python.quickfix.PythonTextEdit.remove;
import static org.sonar.python.quickfix.PythonTextEdit.replaceAt;

@Rule(key = "S1940")
public class BooleanCheckNotInvertedCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use the opposite operator (\"%s\") instead.";

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
        String oppositeOperator = oppositeOperator(binaryExp.operator());
        createQuickFix(ctx.addIssue(original, String.format(MESSAGE, oppositeOperator)), oppositeOperator, binaryExp, original);
      }
    } else if (negatedExpr.is(Kind.IN, Kind.IS)) {
      BinaryExpression isInExpr = (BinaryExpression) negatedExpr;
      String oppositeOperator = oppositeOperator(isInExpr.operator(), isInExpr);
      createQuickFix(ctx.addIssue(original, String.format(MESSAGE, oppositeOperator)), oppositeOperator, isInExpr, original);
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

  private static void createQuickFix(PreciseIssue preciseIssue, String oppositeOperator, BinaryExpression toReplace, Tree toRemove) {
    IssueWithQuickFix issue = (IssueWithQuickFix) preciseIssue;

    PythonTextEdit replaceEdit;
    if (toReplace.is(Kind.IS) && ((IsExpressionImpl) toReplace).notToken() != null) {
        replaceEdit = replaceAt(((IsExpressionImpl) toReplace).notToken(), "");
    } else if (toReplace.is(Kind.IN) && ((InExpressionImpl) toReplace).notToken() != null) {
        replaceEdit = replaceAt(((InExpressionImpl) toReplace).notToken(), "");
    } else {
      replaceEdit = replaceAt(toReplace.operator(), oppositeOperator);
    }

    PythonQuickFix quickFix = PythonQuickFix.newQuickFix(String.format("Use %s instead", oppositeOperator))
      .addTextEdit(replaceEdit, remove(toRemove))
      .build();
    issue.addQuickFix(quickFix);
  }

}
