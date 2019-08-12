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
package org.sonar.python.checks.hotspots;

import com.intellij.psi.PsiElement;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyAssignmentStatement;
import com.jetbrains.python.psi.PyBoolLiteralExpression;
import com.jetbrains.python.psi.PyCallExpression;
import com.jetbrains.python.psi.PyClass;
import com.jetbrains.python.psi.PyExpression;
import com.jetbrains.python.psi.PyFunction;
import com.jetbrains.python.psi.PyKeywordArgument;
import com.jetbrains.python.psi.PyReferenceExpression;
import java.util.Arrays;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.SubscriptionContext;

@Rule(key = "S4507")
public class DebugModeCheck extends PythonCheck {
  private static final String MESSAGE = "Make sure this debug feature is deactivated before delivering the code in production.";
  private static final Set<String> debugProperties = immutableSet("DEBUG", "DEBUG_PROPAGATE_EXCEPTIONS");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.ASSIGNMENT_STATEMENT, ctx -> {
      if (ctx.syntaxNode().getContainingFile().getVirtualFile().getName().equals("global_settings.py")) {
        PyAssignmentStatement node = (PyAssignmentStatement) ctx.syntaxNode();
        PyExpression lhs = node.getLeftHandSideExpression();
        if (lhs != null && debugProperties.contains(lhs.getText()) && node.getAssignedValue() instanceof PyBoolLiteralExpression) {
          PyBoolLiteralExpression assignedValue = (PyBoolLiteralExpression) node.getAssignedValue();
          if (assignedValue.getValue()) {
            ctx.addIssue(node, MESSAGE);
          }
        }
      }

    });

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
            if ("django.conf.LazySettings.configure".equals(qualifiedName + "." + pyFunction.getName())) {
              checkDebugArgument(ctx, node);
            }
          }
        }
      }
    });

  }

  private void checkDebugArgument(SubscriptionContext ctx, PyCallExpression node) {
    Arrays.stream(node.getArguments()).forEach(arg -> {
      if (arg instanceof PyKeywordArgument) {
        PyKeywordArgument pyKeywordArgument = (PyKeywordArgument) arg;
        if (debugProperties.contains(pyKeywordArgument.getKeyword())) {
          if (pyKeywordArgument.getValueExpression() instanceof PyBoolLiteralExpression) {
            PyBoolLiteralExpression valueExpression = (PyBoolLiteralExpression) pyKeywordArgument.getValueExpression();
            if (valueExpression.getValue()) {
              ctx.addIssue(pyKeywordArgument, MESSAGE);
            }
          }
        }
      }
    });
  }

}
