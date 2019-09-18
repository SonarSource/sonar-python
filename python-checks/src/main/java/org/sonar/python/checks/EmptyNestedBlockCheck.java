/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import com.sonar.sslr.api.Trivia;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;

@Rule(key = EmptyNestedBlockCheck.CHECK_KEY)
public class EmptyNestedBlockCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S108";
  private static final String MESSAGE = "Either remove or fill this block of code.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.STATEMENT_LIST, ctx -> {
      PyStatementListTree statementListTree = (PyStatementListTree) ctx.syntaxNode();
      List<PyStatementTree> nonPassStatements = statementListTree.statements().stream()
        .filter(stmt -> !stmt.is(Kind.PASS_STMT))
        .collect(Collectors.toList());
      if (!nonPassStatements.isEmpty()) {
        return;
      }
      Tree parent = statementListTree.parent();
      if (parent.is(Kind.FUNCDEF) || parent.is(Kind.CLASSDEF) || parent.is(Kind.EXCEPT_CLAUSE)) {
        return;
      }
      if (!containsComment(statementListTree.tokens())) {
        if (statementListTree.statements().isEmpty()) {
          ctx.addIssue(statementListTree.firstToken(), MESSAGE);
        } else {
          ctx.addIssue(statementListTree.statements().get(0), MESSAGE);
        }
      }
    });
  }

  private static boolean containsComment(List<PyToken> tokens) {
    for (PyToken token : tokens) {
      for (Trivia trivia : token.trivia()) {
        if (trivia.isComment()) {
          return true;
        }
      }
    }
    return false;
  }
}
