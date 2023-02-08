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
package org.sonar.python.checks.tests;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;

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

        var issue = (IssueWithQuickFix) ctx.addIssue(tuple, MESSAGE);

        if (tuple.leftParenthesis() != null && tuple.rightParenthesis() != null) {
          // defensive condition
          issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
            .addTextEdit(PythonTextEdit.remove(tuple.leftParenthesis()))
            .addTextEdit(PythonTextEdit.remove(tuple.rightParenthesis()))
            .build());
        }
      }
    });
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
