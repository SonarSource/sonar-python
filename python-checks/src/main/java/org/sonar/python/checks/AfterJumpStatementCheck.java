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

import com.intellij.lang.ASTNode;
import com.intellij.psi.tree.IElementType;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyFile;
import com.jetbrains.python.psi.PyFileElementType;
import com.jetbrains.python.psi.PyStatement;
import com.jetbrains.python.psi.PyStatementList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.SubscriptionContext;

@Rule(key = "S1763")
public class AfterJumpStatementCheck extends PythonCheck {

  private static final Set<IElementType> JUMP_TYPES = new HashSet<>(Arrays.asList(
    PyElementTypes.RETURN_STATEMENT,
    PyElementTypes.RAISE_STATEMENT,
    PyElementTypes.BREAK_STATEMENT,
    PyElementTypes.CONTINUE_STATEMENT
  ));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.STATEMENT_LIST, ctx -> {
      PyStatementList statementList = (PyStatementList) ctx.syntaxNode();
      checkStatements(ctx, Arrays.asList(statementList.getStatements()));
    });
    context.registerSyntaxNodeConsumer(PyFileElementType.INSTANCE, ctx -> {
      PyFile pyFile = (PyFile) ctx.syntaxNode();
      checkStatements(ctx, pyFile.getStatements());
    });
  }

  private static void checkStatements(SubscriptionContext ctx, List<PyStatement> statements) {
    for (PyStatement statement : statements.subList(0, Math.max(statements.size() - 1, 0))) {
      if (JUMP_TYPES.contains(statement.getNode().getElementType())) {
        ASTNode keyword = statement.getNode().findLeafElementAt(0);
        ctx.addIssue(keyword.getPsi(), String.format(
          "Refactor this piece of code to not have any dead code after this \"%s\".", keyword.getText()));
      }
    }
  }

}

