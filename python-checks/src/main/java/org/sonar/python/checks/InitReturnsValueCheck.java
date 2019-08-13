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
import com.intellij.psi.util.PsiTreeUtil;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyElement;
import com.jetbrains.python.psi.PyExpression;
import com.jetbrains.python.psi.PyFunction;
import com.jetbrains.python.psi.PyReturnStatement;
import com.jetbrains.python.psi.PyYieldExpression;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;

@Rule(key = InitReturnsValueCheck.CHECK_KEY)
public class InitReturnsValueCheck extends PythonCheck {

  private static final String MESSAGE_RETURN = "Remove this return value.";
  private static final String MESSAGE_YIELD = "Remove this yield statement.";

  public static final String CHECK_KEY = "S2734";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.FUNCTION_DECLARATION, ctx -> {
      PyFunction function = (PyFunction) ctx.syntaxNode();
      ASTNode functionNameNode = function.getNameNode();
      if (functionNameNode == null || !"__init__".equals(functionNameNode.getText())) {
        return;
      }
      for (PyElement returnOrYield : PsiTreeUtil.findChildrenOfAnyType(function, PyReturnStatement.class, PyYieldExpression.class)) {
        PyFunction returnOrYieldFunction = PsiTreeUtil.getParentOfType(returnOrYield, PyFunction.class);
        if (returnOrYieldFunction == function && !isReturnNone(returnOrYield)) {
          String message = returnOrYield instanceof PyYieldExpression ? MESSAGE_YIELD : MESSAGE_RETURN;
          ctx.addIssue(returnOrYield, message);
        }
      }
    });
  }

  private static boolean isReturnNone(PyElement element) {
    if (element instanceof PyReturnStatement) {
      PyExpression expression = ((PyReturnStatement) element).getExpression();
      return expression == null || expression.getNode().getElementType() == PyElementTypes.NONE_LITERAL_EXPRESSION;
    }
    return false;
  }
}

