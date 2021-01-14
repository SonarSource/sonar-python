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

@Rule(key = ReturnYieldOutsideFunctionCheck.CHECK_KEY)
public class ReturnYieldOutsideFunctionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this use of \"%s\".";
  public static final String CHECK_KEY = "S2711";
  private static final Consumer<SubscriptionContext> SUBSCRIPTION_CONTEXT_CONSUMER = ctx -> {
    Statement returnStatement = (Statement) ctx.syntaxNode();
    Tree currentParent = returnStatement.parent();
    while (currentParent != null) {
      if (currentParent.is(Tree.Kind.FUNCDEF)) {
        return;
      } else if (currentParent.is(Tree.Kind.CLASSDEF)) {
        ctx.addIssue(returnStatement, String.format(MESSAGE, returnStatement.firstToken().value()));
        return;
      }
      currentParent = currentParent.parent();
    }
    ctx.addIssue(returnStatement, String.format(MESSAGE, returnStatement.firstToken().value()));
  };

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.YIELD_STMT, SUBSCRIPTION_CONTEXT_CONSUMER);
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_STMT, SUBSCRIPTION_CONTEXT_CONSUMER);
  }
}
