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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyReturnStatementTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.PyYieldStatementTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.tree.BaseTreeVisitor;

@Rule(key = TooManyReturnsCheck.CHECK_KEY)
public class TooManyReturnsCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S1142";

  private static final int DEFAULT_MAX = 3;
  private static final String MESSAGE = "This function has %s returns or yields, which is more than the %s allowed.";

  @RuleProperty(key = "max", defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      PyFunctionDefTree func = ((PyFunctionDefTree) ctx.syntaxNode());
      ReturnCountVisitor returnCountVisitor = new ReturnCountVisitor();
      func.body().accept(returnCountVisitor);

      if (returnCountVisitor.returnStatements.size() > max) {
        PreciseIssue preciseIssue = ctx.addIssue(func.name(), String.format(MESSAGE, returnCountVisitor.returnStatements.size(), max));
        returnCountVisitor.returnStatements.forEach(r -> {
          preciseIssue.secondary(r, null);
        });
      }
    });
  }

  private static class ReturnCountVisitor extends BaseTreeVisitor {

    private List<PyStatementTree> returnStatements = new ArrayList<>();

    @Override
    public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
    }

    @Override
    public void visitReturnStatement(PyReturnStatementTree pyReturnStatementTree) {
      super.visitReturnStatement(pyReturnStatementTree);
      returnStatements.add(pyReturnStatementTree);
    }

    @Override
    public void visitYieldStatement(PyYieldStatementTree pyYieldStatementTree) {
      super.visitYieldStatement(pyYieldStatementTree);
      returnStatements.add(pyYieldStatementTree);
    }
  }
}
