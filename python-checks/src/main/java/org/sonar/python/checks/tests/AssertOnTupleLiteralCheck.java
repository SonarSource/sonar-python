/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks.tests;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S5905")
public class AssertOnTupleLiteralCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Fix this assertion on a tuple literal.";
  public static final String QUICK_FIX_MESSAGE = "Remove parentheses";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> {
      AssertStatement assertStatement = (AssertStatement) ctx.syntaxNode();

      if (assertStatement.condition().is(Tree.Kind.TUPLE) && assertStatement.message() == null) {
        var tuple = (Tuple) assertStatement.condition();

        var issue = ctx.addIssue(tuple, MESSAGE);

        if (isSingletonTupleWithParenthesis(tuple)) {
          Token comma =  tuple.commas().get(0);
          var quickfixBuilder = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
            .addTextEdit(TextEditUtils.remove(tuple.leftParenthesis()))
            .addTextEdit(TextEditUtils.replaceRange(comma, tuple.rightParenthesis(), ""));

          issue.addQuickFix(quickfixBuilder.build());
        }
      }
    });
  }

  private static boolean isSingletonTupleWithParenthesis(Tuple tuple) {
    return tuple.leftParenthesis() != null && tuple.rightParenthesis() != null && tuple.elements().size() == 1;
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
