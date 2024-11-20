/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;

@Rule(key = "S1143")
public class JumpInFinallyCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.BREAK_STMT, ctx -> checkJumpStatement(ctx, "break"));
    context.registerSyntaxNodeConsumer(Kind.CONTINUE_STMT, ctx -> checkJumpStatement(ctx, "continue"));
    context.registerSyntaxNodeConsumer(Kind.RETURN_STMT, ctx -> checkJumpStatement(ctx, "return"));
  }

  private static void checkJumpStatement(SubscriptionContext ctx, String keyword) {
    Tree tree = ctx.syntaxNode();
    if (isInFinally(tree)) {
      ctx.addIssue(tree, String.format("Remove this \"%s\" statement from this \"finally\" block.", keyword));
    }
  }

  private static boolean isInFinally(Tree tree) {
    Tree parent = tree.parent();
    while (parent != null) {
      if (parent.is(Kind.FINALLY_CLAUSE)) {
        return true;
      }
      if (parent.is(Kind.FUNCDEF)) {
        return false;
      }
      if (tree.is(Kind.BREAK_STMT, Kind.CONTINUE_STMT) && parent.is(Kind.FOR_STMT, Kind.WHILE_STMT)) {
        return false;
      }
      parent = parent.parent();
    }
    return false;
  }

}
