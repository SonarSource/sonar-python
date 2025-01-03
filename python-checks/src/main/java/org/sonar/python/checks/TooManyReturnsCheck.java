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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;

@Rule(key = TooManyReturnsCheck.CHECK_KEY)
public class TooManyReturnsCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S1142";

  private static final int DEFAULT_MAX = 3;
  private static final String MESSAGE = "This function has %s returns or yields, which is more than the %s allowed.";

  @RuleProperty(
    key = "max",
    description = "Maximum allowed return statements per function",
    defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef func = ((FunctionDef) ctx.syntaxNode());
      ReturnCountVisitor returnCountVisitor = new ReturnCountVisitor();
      func.body().accept(returnCountVisitor);

      if (returnCountVisitor.returnStatements.size() > max) {
        PreciseIssue preciseIssue = ctx.addIssue(func.name(), String.format(MESSAGE, returnCountVisitor.returnStatements.size(), max));
        returnCountVisitor.returnStatements.forEach(r -> preciseIssue.secondary(r, "\"return\" statement."));
      }
    });
  }

  private static class ReturnCountVisitor extends BaseTreeVisitor {

    private List<Statement> returnStatements = new ArrayList<>();

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      // Ignore nested function definitions
    }

    @Override
    public void visitReturnStatement(ReturnStatement pyReturnStatementTree) {
      super.visitReturnStatement(pyReturnStatementTree);
      returnStatements.add(pyReturnStatementTree);
    }

    @Override
    public void visitYieldStatement(YieldStatement pyYieldStatementTree) {
      super.visitYieldStatement(pyYieldStatementTree);
      returnStatements.add(pyYieldStatementTree);
    }
  }
}
