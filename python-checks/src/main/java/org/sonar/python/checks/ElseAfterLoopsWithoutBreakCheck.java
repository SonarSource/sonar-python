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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WhileStatement;

// https://jira.sonarsource.com/browse/RSPEC-2836
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
    ElseClause elseClause = getElseClause(loop);

    if ((elseClause != null) && !containsFreeBreak(getLoopBody(loop))) {
      subscriptionContext.addIssue(elseClause.elseKeyword(), MESSAGE);
    }
  }

  private static ElseClause getElseClause(Tree loop) {
    return loop.is(Tree.Kind.FOR_STMT) ? ((ForStatement) loop).elseClause() : ((WhileStatement) loop).elseClause();
  }

  private static Tree getLoopBody(Tree loop) {
    return loop.is(Tree.Kind.FOR_STMT) ? ((ForStatement) loop).body() : ((WhileStatement) loop).body();
  }

  /**
   * Checks whether a tree has a "free" break (meaning that it is not bound by any loop, analogously to "free variable")
   */
  private static boolean containsFreeBreak(Tree t) {
    if (t.is(Tree.Kind.BREAK_STMT)) {
      return true;
    } else if (t.is(Tree.Kind.WHILE_STMT, Tree.Kind.FOR_STMT)) {
      // All breaks in the bodies of the loops are bound, check only the `else`-clause, if present
      ElseClause elseClause = getElseClause(t);
      if (elseClause != null) {
        return containsFreeBreak(elseClause);
      } else {
        return false;
      }
    } else {
      return t.children().stream().anyMatch(ElseAfterLoopsWithoutBreakCheck::containsFreeBreak);
    }
  }
}
