/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import org.sonar.plugins.python.api.tree.BreakStatement;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.python.tree.TreeUtils;

// https://jira.sonarsource.com/browse/RSPEC-2836
// https://jira.sonarsource.com/browse/SONARPY-650
@Rule(key = "S2836")
public class ElseAfterLoopsWithoutBreakCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Add a \"break\" statement or remove this \"else\" clause.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, ElseAfterLoopsWithoutBreakCheck::check);
    context.registerSyntaxNodeConsumer(Tree.Kind.WHILE_STMT, ElseAfterLoopsWithoutBreakCheck::check);
  }

  private static void check(SubscriptionContext subscriptionContext) {
    Tree loop = subscriptionContext.syntaxNode();
    ElseClause elseClause = loop.is(Tree.Kind.FOR_STMT) ?
      ((ForStatement) loop).elseClause() :
      ((WhileStatement) loop).elseClause();

    if ((elseClause != null) &&
      !TreeUtils.hasDescendant(loop, t -> t.is(Tree.Kind.BREAK_STMT) && (brokenLoop((BreakStatement) t) == loop))) {

      subscriptionContext.addIssue(elseClause.elseKeyword(), MESSAGE);
    }
  }

  private static Tree brokenLoop(BreakStatement stmt) {
    return TreeUtils.firstAncestorOfKind(stmt, Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT);
  }
}
