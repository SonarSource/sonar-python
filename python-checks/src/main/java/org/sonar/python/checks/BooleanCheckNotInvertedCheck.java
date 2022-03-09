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
    if (negatedExpr.is(Tree.Kind.COMPARISON)) {
      BinaryExpression binaryExp = (BinaryExpression) negatedExpr;
      if(!(binaryExp.leftOperand().getKind().equals(Tree.Kind.COMPARISON)  || binaryExp.rightOperand().getKind().equals(Tree.Kind.COMPARISON))) {
        ctx.addIssue(original, String.format(MESSAGE, oppositeOperator(((BinaryExpression) negatedExpr).operator())));
      }
    }
  }

  private static String oppositeOperator(Token operator){
    String s;
    switch (operator.value()){
      case ">"  :
        s = "<=";
        break;
      case ">=" :
        s = "<";
        break;
      case "<"  :
        s = ">=";
        break;
      case "<=" :
        s = ">";
        break;
      case "==" :
        s = "!=";
        break;
      case "!=" :
        s = "==";
        break;
      default   :
        s = "unknown";
        break;
    }
    return s;
  }

}
