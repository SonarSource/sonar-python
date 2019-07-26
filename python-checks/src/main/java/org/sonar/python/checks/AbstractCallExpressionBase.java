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
import com.jetbrains.python.psi.PyCallExpression;
import com.jetbrains.python.psi.PyClass;
import com.jetbrains.python.psi.PyExpression;
import com.jetbrains.python.psi.PyFunction;
import com.jetbrains.python.psi.PyReferenceExpression;
import java.util.Set;
import org.sonar.python.PythonCheck;

public abstract class AbstractCallExpressionBase extends PythonCheck {

  protected abstract Set<String> functionsToCheck();

  protected abstract String message();

  protected boolean isException(PsiElement node) {
    return false;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.CALL_EXPRESSION, ctx -> {
      PyCallExpression node = (PyCallExpression) ctx.syntaxNode();
      PyExpression callee = node.getCallee();
      if (callee instanceof PyReferenceExpression) {
        PsiElement resolve = ((PyReferenceExpression) callee).getReference().resolve();
        if (resolve instanceof PyFunction) {
          PyFunction pyFunction = (PyFunction) resolve;
          PyClass containingClass = pyFunction.getContainingClass();
          if (containingClass != null) {
            String qualifiedName = containingClass.getQualifiedName();
            if (!isException(node) && functionsToCheck().contains(qualifiedName + "." + pyFunction.getName())) {
              ctx.addIssue(node, message());
            }
          }
        }
      }
    });
  }
}
