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
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5704")
public class BareRaiseInFinallyCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.RAISE_STMT, ctx -> {
      RaiseStatement raiseStatement = (RaiseStatement) ctx.syntaxNode();
      if (!raiseStatement.expressions().isEmpty()) {
        return;
      }
      if (isWithinExceptClause(raiseStatement) || isWithinExitFunction(raiseStatement)) {
        return;
      }
      Tree parent = TreeUtils.firstAncestorOfKind(raiseStatement, Tree.Kind.FINALLY_CLAUSE, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF);
      if (parent != null && parent.is(Tree.Kind.FINALLY_CLAUSE)) {
        ctx.addIssue(raiseStatement, "Refactor this code so that any active exception raises naturally.");
      }
    });
  }

  private static boolean isWithinExceptClause(RaiseStatement raiseStatement) {
    Tree tree = TreeUtils.firstAncestorOfKind(raiseStatement, Tree.Kind.EXCEPT_CLAUSE, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF);
    return tree != null && tree.is(Tree.Kind.EXCEPT_CLAUSE);
  }

  private static boolean isWithinExitFunction(RaiseStatement finallyClause) {
    return TreeUtils.firstAncestor(finallyClause, t -> t.is(Tree.Kind.FUNCDEF)
      && ((FunctionDef) t).name().name().equals("__exit__")) != null;
  }
}
