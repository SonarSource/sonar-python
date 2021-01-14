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
