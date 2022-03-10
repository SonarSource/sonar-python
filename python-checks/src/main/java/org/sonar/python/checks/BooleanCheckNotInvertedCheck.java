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
import org.sonar.plugins.python.api.tree.UnaryExpression;

@Rule(key = "S1940")
public class BooleanCheckNotInvertedCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use the opposite operator (\"%s\") instead.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.NOT, ctx -> checkNotOutsideParentheses(ctx, (UnaryExpression) ctx.syntaxNode()));
  }

  private static void checkNotOutsideParentheses(SubscriptionContext ctx, UnaryExpression original) {
    Expression negatedExpr = original.expression();
    while (negatedExpr.is(Tree.Kind.PARENTHESIZED)) {
      negatedExpr = ((ParenthesizedExpression) negatedExpr).expression();
    }
    if(negatedExpr.is(Tree.Kind.COMPARISON)) {
      BinaryExpression binaryExp = (BinaryExpression) negatedExpr;
      // Don't raise warning with "not a == b == c" because a == b != c is not equivalent
      if(!binaryExp.leftOperand().is(Tree.Kind.COMPARISON)) {
        ctx.addIssue(original, String.format(MESSAGE, oppositeOperator(binaryExp.operator())));
      }
    } else if(negatedExpr.is(Tree.Kind.IN) || negatedExpr.is(Tree.Kind.IS) ) {
      BinaryExpression isInExpr = (BinaryExpression) negatedExpr;
      ctx.addIssue(original, String.format(MESSAGE, oppositeOperator(isInExpr.operator(), isInExpr)));
    }
  }

  private static String oppositeOperator(Token operator){
    return oppositeOperatorString(operator.value());
  }

  private static String oppositeOperator(Token operator, Expression expr){
    String s = operator.value();
    if(expr.is(Tree.Kind.IS) && ((IsExpression) expr).notToken() != null){
      s = s + " not";
    } else if(expr.is(Tree.Kind.IN) && ((InExpression) expr).notToken() != null){
      s = "not " + s;
    }
    return oppositeOperatorString(s);
  }

  static String oppositeOperatorString(String stringOperator){
    switch (stringOperator){
      case ">"  :
        return "<=";
      case ">=" :
        return "<";
      case "<"  :
        return ">=";
      case "<=" :
        return ">";
      case "==" :
        return "!=";
      case "!=" :
        return "==";
      case "is" :
        return "is not";
      case "is not":
        return "is";
      case "in" :
        return "not in";
      case "not in":
        return "in";
      default   :
        throw new IllegalArgumentException("Unknown comparison operator : " + stringOperator);
    }
  }
}
