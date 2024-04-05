/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.v2.PythonType;

@Rule(key = "S5890")
public class InconsistentTypeHintCheck extends PythonSubscriptionCheck {

  private static final String NONE_TYPE_NAME = "NoneType";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, ctx -> {
      AnnotatedAssignment annotatedAssignment = (AnnotatedAssignment) ctx.syntaxNode();
      Expression assignedExpression = annotatedAssignment.assignedValue();
      if (assignedExpression == null) {
        return;
      }
      checkAnnotatedAssignment(ctx, annotatedAssignment, assignedExpression);
    });
  }

  private static void checkAnnotatedAssignment(SubscriptionContext ctx, AnnotatedAssignment annotatedAssignment, Expression assignedExpression) {
    PythonType assignedType = assignedExpression.pythonType();
    PythonType hintType = annotatedAssignment.variable().pythonType();
    if (!assignedType.isCompatibleWith(hintType)) {
      String message = getIssueMessage(annotatedAssignment.variable(), assignedType, hintType);

      ctx.addIssue(assignedExpression, message)
        .secondary(annotatedAssignment.annotation().expression(), null);
    }
  }

  private static String getIssueMessage(Expression variable, PythonType inferredType, PythonType expectedType) {
    String expectedTypeName = expectedType.displayName();
    String inferredTypeName = inferredType.displayName();

    var variableName = Optional.ofNullable(TreeUtils.nameFromExpression(variable))
      .map(name -> "\"" + name + "\"")
      .orElse("this expression");

    if (NONE_TYPE_NAME.equals(inferredTypeName)) {
      return String.format("Replace the type hint \"%1$s\" with \"Optional[%1s]\" or don't assign \"None\" to %2$s",
        expectedTypeName,
        variableName
      );
    } else {
      String inferredTypeNameMessage = inferredTypeName != null ? String.format(" instead of \"%s\"", inferredTypeName) : "";
      return String.format("Assign to %s a value of type \"%s\"%s or update its type hint.",
        variableName,
        expectedTypeName,
        inferredTypeNameMessage);
    }
  }
}
