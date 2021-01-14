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

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Tree;

/**
 * Note that implementation differs from AbstractOneStatementPerLineCheck due to Python specifics
 */
@Rule(key = OneStatementPerLineCheck.CHECK_KEY)
public class OneStatementPerLineCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "OneStatementPerLine";
  private final Map<Integer, Integer> statementsPerLine = new HashMap<>();
  private SubscriptionContext subscriptionContext;
  private static final List<Tree.Kind> kinds = Arrays.asList(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.COMPOUND_ASSIGNMENT, Tree.Kind.EXPRESSION_STMT, Tree.Kind.IMPORT_STMT,
    Tree.Kind.IMPORT_NAME, Tree.Kind.IMPORT_FROM, Tree.Kind.CONTINUE_STMT, Tree.Kind.BREAK_STMT, Tree.Kind.YIELD_STMT, Tree.Kind.RETURN_STMT, Tree.Kind.PRINT_STMT,
    Tree.Kind.PASS_STMT, Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT, Tree.Kind.IF_STMT, Tree.Kind.ELSE_CLAUSE, Tree.Kind.RAISE_STMT, Tree.Kind.TRY_STMT, Tree.Kind.EXCEPT_CLAUSE,
    Tree.Kind.EXEC_STMT, Tree.Kind.ASSERT_STMT, Tree.Kind.DEL_STMT, Tree.Kind.GLOBAL_STMT, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      statementsPerLine.clear();
      this.subscriptionContext = ctx;
    });

    kinds.forEach(k -> context.registerSyntaxNodeConsumer(k, this::checkStatement));
  }

  private void checkStatement(SubscriptionContext ctx) {
    int line = ctx.syntaxNode().firstToken().line();
    if (!statementsPerLine.containsKey(line)) {
      statementsPerLine.put(line, 0);
    }
    statementsPerLine.put(line, statementsPerLine.get(line) + 1);
  }

  @Override
  public void leaveFile() {
    for (Map.Entry<Integer, Integer> statementsAtLine : statementsPerLine.entrySet()) {
      if (statementsAtLine.getValue() > 1) {
        String message = String.format("At most one statement is allowed per line, but %s statements were found on this line.", statementsAtLine.getValue());
        int lineNumber = statementsAtLine.getKey();
        subscriptionContext.addLineIssue(message, lineNumber);
      }
    }
  }
}
