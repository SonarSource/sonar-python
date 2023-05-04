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

import java.util.List;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.EXPRESSION_STMT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.PASS_STMT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.STATEMENT_LIST;
import static org.sonar.plugins.python.api.tree.Tree.Kind.STRING_LITERAL;

@Rule(key = "S2772")
public class NeedlessPassCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this unneeded \"pass\".";
  public static final String QUICK_FIX_MESSAGE = "Remove the \"pass\" statement";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(STATEMENT_LIST, ctx -> {
      List<Statement> statements = ((StatementList) ctx.syntaxNode()).statements().stream()
        .filter(NeedlessPassCheck::isNotStringLiteralExpressionStatement)
        .collect(Collectors.toList());
      if (statements.size() <= 1) {
        return;
      }
      statements.stream()
        .filter(st -> st.is(PASS_STMT))
        .findFirst()
        .ifPresent(st -> {
          var textEdit = createRemoveStatementTextEdit(statements, st);
          var quickFix = PythonQuickFix
            .newQuickFix(QUICK_FIX_MESSAGE)
            .addTextEdit(textEdit)
            .build();
          ctx.addIssue(st, MESSAGE).addQuickFix(quickFix);
        });
    });
  }

  private static PythonTextEdit createRemoveStatementTextEdit(List<Statement> statements, Statement toRemove) {
    var removeIndex = statements.indexOf(toRemove);
    var last = removeIndex == statements.size() - 1;
    if (last) {
      var previous = statements.get(removeIndex - 1);
      var removeFrom = TreeUtils.getTreeSeparatorOrLastToken(previous);
      var removeTo = TreeUtils.getTreeSeparatorOrLastToken(toRemove);
      return TextEditUtils.removeRange(
        removeFrom.line(),
        removeFrom.column(),
        removeTo.line(),
        removeTo.column());
    } else {
      return TextEditUtils.removeUntil(toRemove, statements.get(removeIndex + 1));
    }
  }

  private static boolean isNotStringLiteralExpressionStatement(Statement st) {
    return !(st.is(EXPRESSION_STMT) && ((ExpressionStatement) st).expressions().stream().allMatch(e -> e.is(STRING_LITERAL)));
  }
}

