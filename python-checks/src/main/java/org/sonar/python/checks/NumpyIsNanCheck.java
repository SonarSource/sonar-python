/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6725")
public class NumpyIsNanCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.COMPARISON, ctx -> checkForIsNan(ctx));
  }

  private static void checkForIsNan(SubscriptionContext ctx) {
    BinaryExpression be = (BinaryExpression) ctx.syntaxNode();
    String value = be.operator().value();
    if (!("==".equals(value) || "!=".equals(value))) {
      return;
    }
    // What if we have expressions that evaluate to numpy.nan?
    checkOperand(ctx, be.leftOperand(), be);
    checkOperand(ctx, be.rightOperand(), be);
  }

  private static void checkOperand(SubscriptionContext ctx, Expression operand, BinaryExpression be) {
    if (operand.is(Tree.Kind.QUALIFIED_EXPR)) {
      QualifiedExpression expression = (QualifiedExpression) operand;
      Symbol symbol = expression.symbol();
      System.out.println(symbol);
      if (symbol != null && "numpy.nan".equals(symbol.fullyQualifiedName())) {
        ctx.addIssue(be, "Equality checks should not be made against \"numpy.nan\". Use numpy.isnan() instead.");
      }
    }
  }
}
