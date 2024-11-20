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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
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
