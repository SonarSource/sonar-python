/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.tree.Tree.Kind;

@Rule(key = PreIncrementDecrementCheck.CHECK_KEY)
public class PreIncrementDecrementCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "PreIncrementDecrement";
  private static final String MESSAGE = "This statement doesn't produce the expected result, replace use of non-existent pre-%srement operator";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.UNARY_PLUS, PreIncrementDecrementCheck::checkIncrementDecrement);
    context.registerSyntaxNodeConsumer(Kind.UNARY_MINUS, PreIncrementDecrementCheck::checkIncrementDecrement);
  }

  private static void checkIncrementDecrement(SubscriptionContext ctx) {
    UnaryExpression unaryExpressionTree = (UnaryExpression) ctx.syntaxNode();
    Kind kind = unaryExpressionTree.getKind();
    if (unaryExpressionTree.expression().is(kind)) {
      ctx.addIssue(unaryExpressionTree, String.format(MESSAGE, kind == Kind.UNARY_PLUS ? "inc" : "dec"));
    }
  }
}
