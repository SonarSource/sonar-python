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

import com.intellij.psi.PsiElement;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyIfPart;
import com.jetbrains.python.psi.PyIfStatement;
import com.jetbrains.python.psi.PyStatement;
import com.jetbrains.python.psi.PyStatementList;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;

@Rule(key = "S1066")
public class CollapsibleIfStatementsCheck extends PythonCheck {
  private static final String MESSAGE = "Merge this if statement with the enclosing one.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.IF_STATEMENT, ctx -> {
      PyIfStatement ifStatement = (PyIfStatement) ctx.syntaxNode();

      if (ifStatement.getElsePart() != null) {
        return;
      }

      PyIfPart[] elifParts = ifStatement.getElifParts();
      PyIfPart lastIfPart = elifParts.length == 0 ? ifStatement.getIfPart() : elifParts[elifParts.length - 1];

      PyIfStatement singleIfChild = singleIfChild(lastIfPart.getStatementList());
      if (singleIfChild != null && singleIfChild.getElifParts().length == 0 && singleIfChild.getElsePart() == null) {
        ctx.addIssue(ifKeyword(singleIfChild), MESSAGE)
          .secondary(ifKeyword(ifStatement), "enclosing");
      }
    });
  }

  @CheckForNull
  private static PyIfStatement singleIfChild(PyStatementList statementList) {
    PyStatement[] statements = statementList.getStatements();
    if (statements.length == 1) {
      PyStatement statement = statements[0];
      if (statement.getNode().getElementType() == PyElementTypes.IF_STATEMENT) {
        return (PyIfStatement) statement;
      }
    }
    return null;
  }

  private static PsiElement ifKeyword(PyIfStatement ifStatement) {
    return ifStatement.getNode().findLeafElementAt(0).getPsi();
  }
}
