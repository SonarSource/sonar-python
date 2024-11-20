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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;

@Rule(key = InitReturnsValueCheck.CHECK_KEY)
public class InitReturnsValueCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_RETURN = "Remove this return value.";
  private static final String MESSAGE_YIELD = "Remove this yield statement.";

  public static final String CHECK_KEY = "S2734";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef func = (FunctionDef) ctx.syntaxNode();
      if (!"__init__".equals(func.name().name())) {
        return;
      }
      ReturnAndYieldVisitor returnAndYieldVisitor = new ReturnAndYieldVisitor();
      func.body().accept(returnAndYieldVisitor);
      for (Tree returnNode : returnAndYieldVisitor.returnNodes) {
        String message = returnNode.is(Tree.Kind.RETURN_STMT) ? MESSAGE_RETURN : MESSAGE_YIELD;
        ctx.addIssue(returnNode, message);
      }
    });
  }

  private static class ReturnAndYieldVisitor extends BaseTreeVisitor {

    List<Tree> returnNodes = new ArrayList<>();

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      // Ignore nested function definitions
    }

    @Override
    public void visitReturnStatement(ReturnStatement pyReturnStatementTree) {
      if (pyReturnStatementTree.expressions().isEmpty() || pyReturnStatementTree.expressions().get(0).is(Tree.Kind.NONE)) {
        return;
      }
      returnNodes.add(pyReturnStatementTree);
    }

    @Override
    public void visitYieldStatement(YieldStatement pyYieldStatementTree) {
      returnNodes.add(pyYieldStatementTree);
    }
  }
}
