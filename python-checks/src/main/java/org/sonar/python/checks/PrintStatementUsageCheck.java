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

import com.intellij.psi.util.PsiTreeUtil;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyElement;
import com.jetbrains.python.psi.PyExpression;
import com.jetbrains.python.psi.PyParenthesizedExpression;
import com.jetbrains.python.psi.PyPrintTarget;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;

@Rule(key = "PrintStatementUsage")
public class PrintStatementUsageCheck extends PythonCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.PRINT_STATEMENT, ctx -> {
      List<PyElement> expressions = PsiTreeUtil.getChildrenOfAnyType(ctx.syntaxNode(), PyExpression.class, PyPrintTarget.class);
      if (expressions.size() == 1 && expressions.get(0) instanceof PyParenthesizedExpression) {
        return;
      }
      ctx.addIssue(ctx.syntaxNode().getFirstChild(), "Replace print statement by built-in function.");
    });
  }

}
