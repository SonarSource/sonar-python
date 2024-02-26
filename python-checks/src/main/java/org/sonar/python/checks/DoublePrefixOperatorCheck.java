/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S2761")

public class DoublePrefixOperatorCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use the \"%s\" operator just once or not at all.";
  private static final String MESSAGE_NOT = "Use the \"bool()\" builtin function instead of calling \"not\" twice.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.BITWISE_COMPLEMENT, ctx -> doubleInversionCheck(ctx, (UnaryExpression) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.NOT, ctx -> doubleInversionCheck(ctx, (UnaryExpression) ctx.syntaxNode()));
  }

  private static void doubleInversionCheck(SubscriptionContext ctx, UnaryExpression original) {
    Expression invertedExpr = original.expression();
    boolean doubleInversionFollowed = true;
    while (invertedExpr.is(Tree.Kind.PARENTHESIZED)) {
      doubleInversionFollowed = false;
      invertedExpr = ((ParenthesizedExpression) invertedExpr).expression();
    }

    if (invertedExpr.is(Tree.Kind.NOT, Tree.Kind.BITWISE_COMPLEMENT) && original.is(invertedExpr.getKind())) {
      if (doubleInversionFollowed) {
        // Overloaded __invert__ should not raise any warning
        if (invertedExpr.is(Tree.Kind.NOT)) {
          ctx.addIssue(original, MESSAGE_NOT);
        } else {
          if (((UnaryExpression) invertedExpr).expression().type() == InferredTypes.INT) {
            ctx.addIssue(original, String.format(MESSAGE, original.operator().value()));
          }
        }
      } else {
        ctx.addIssue(original, String.format(MESSAGE, original.operator().value()));
      }
    }
  }
}
