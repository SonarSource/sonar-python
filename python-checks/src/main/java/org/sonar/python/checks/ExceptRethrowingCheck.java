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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.python.checks.utils.CheckUtils;

import static org.sonar.python.checks.utils.Expressions.removeParentheses;

@Rule(key = "S2737")
public class ExceptRethrowingCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add logic to this except clause or eliminate it and rethrow the exception automatically.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TRY_STMT, ctx -> {
      TryStatement tryStatement = (TryStatement) ctx.syntaxNode();
      List<ExceptClause> exceptClauses = tryStatement.exceptClauses();
      if (exceptClauses.isEmpty()) {
        return;
      }
      ExceptClause exceptClause = exceptClauses.get(exceptClauses.size() - 1);
      if (!exceptClause.body().statements().get(0).is(Tree.Kind.RAISE_STMT)) {
        return;
      }
      RaiseStatement raiseStatement = (RaiseStatement) exceptClause.body().statements().get(0);
      if (raiseStatement.expressions().isEmpty()) {
        ctx.addIssue(raiseStatement, MESSAGE);
        return;
      }
      if (exceptClause.exceptionInstance() != null) {
        Expression exceptionInstance = exceptClause.exceptionInstance();
        if (raiseStatement.expressions().size() == 1 && CheckUtils.areEquivalent(removeParentheses(exceptionInstance), removeParentheses(raiseStatement.expressions().get(0)))) {
          ctx.addIssue(raiseStatement, MESSAGE);
        }
      }
    });
  }
}
