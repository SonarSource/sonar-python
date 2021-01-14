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

import java.util.function.Consumer;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = BreakContinueOutsideLoopCheck.CHECK_KEY)
public class BreakContinueOutsideLoopCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this \"%s\" statement";
  public static final String CHECK_KEY = "S1716";

  private static final Consumer<SubscriptionContext> SUBSCRIPTION_CONTEXT_CONSUMER = ctx -> {
    Statement statement = (Statement) ctx.syntaxNode();
    Tree currentParent = statement.parent();
    while (currentParent != null) {
      if (currentParent.is(Tree.Kind.WHILE_STMT) || currentParent.is(Tree.Kind.FOR_STMT)) {
        return;
      } else if (currentParent.is(Tree.Kind.CLASSDEF) || currentParent.is(Tree.Kind.FUNCDEF)) {
        ctx.addIssue(statement, String.format(MESSAGE, statement.firstToken().value()));
        return;
      }
      currentParent = currentParent.parent();
    }
    ctx.addIssue(statement, String.format(MESSAGE, statement.firstToken().value()));
  };

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.BREAK_STMT, SUBSCRIPTION_CONTEXT_CONSUMER);
    context.registerSyntaxNodeConsumer(Tree.Kind.CONTINUE_STMT, SUBSCRIPTION_CONTEXT_CONSUMER);
  }
}

