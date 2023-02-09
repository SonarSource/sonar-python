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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = UselessParenthesisCheck.CHECK_KEY)
public class UselessParenthesisCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "S1110";

  private static final String MESSAGE = "Remove those redundant parentheses.";
  public static final String QUICK_FIX_MESSAGE = "Remove the redundant parentheses";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.PARENTHESIZED, ctx -> {
      ParenthesizedExpression parenthesized = (ParenthesizedExpression) ctx.syntaxNode();
      Expression expression = parenthesized.expression();
      if (expression.is(Tree.Kind.PARENTHESIZED, Tree.Kind.TUPLE, Tree.Kind.GENERATOR_EXPR)) {
        var issue = ctx.addIssue(parenthesized.leftParenthesis(), MESSAGE).secondary(parenthesized.rightParenthesis(), null);
        var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
          .addTextEdit(TextEditUtils.remove(parenthesized.leftParenthesis()), TextEditUtils.remove(parenthesized.rightParenthesis()))
          .build();
        issue.addQuickFix(quickFix);
      }
    });
  }
}
