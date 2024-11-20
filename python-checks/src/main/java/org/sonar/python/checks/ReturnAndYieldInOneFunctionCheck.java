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
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;

@Rule(key = ReturnAndYieldInOneFunctionCheck.CHECK_KEY)
public class ReturnAndYieldInOneFunctionCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "S2712";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef func = ((FunctionDef) ctx.syntaxNode());
      ReturnAndYieldVisitor returnAndYieldVisitor = new ReturnAndYieldVisitor();
      func.body().accept(returnAndYieldVisitor);

      if (returnAndYieldVisitor.hasYield && returnAndYieldVisitor.hasReturn) {
        ctx.addIssue(func.name(), "Use only \"return\" or only \"yield\", not both.");
      }
    });
  }

  private static class ReturnAndYieldVisitor extends BaseTreeVisitor {

    private boolean hasYield = false;
    private boolean hasReturn = false;

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      // Ignore nested function definitions
    }

    @Override
    public void visitReturnStatement(ReturnStatement pyReturnStatementTree) {
      if(!pyReturnStatementTree.expressions().isEmpty()) {
        hasReturn = true;
      }
    }

    @Override
    public void visitYieldStatement(YieldStatement pyYieldStatementTree) {
      hasYield = true;
    }
  }
}
