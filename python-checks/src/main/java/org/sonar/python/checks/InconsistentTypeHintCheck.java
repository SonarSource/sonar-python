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
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeShed;

@Rule(key = "S5890")
public class InconsistentTypeHintCheck extends PythonSubscriptionCheck {

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
    InferredType inferredType = assignedExpression.type();
    TypeAnnotation annotation = annotatedAssignment.annotation();
    InferredType expectedType = InferredTypes.fromTypeAnnotation(annotation);
    if (expectedType.mustBeOrExtend("typing.TypedDict")) {
      // Avoid FPs for TypedDict
      return;
    }
    if (!inferredType.isCompatibleWith(expectedType) || isTypeUsedInsteadOfInstance(assignedExpression, expectedType)) {
      String inferredTypeName = InferredTypes.typeName(inferredType);
      String inferredTypeNameMessage = inferredTypeName != null ? String.format(" instead of \"%s\"", inferredTypeName) : "";
      String nameFromExpression = TreeUtils.nameFromExpression(annotatedAssignment.variable());
      String variableMessage = nameFromExpression != null ? String.format("\"%s\"", nameFromExpression) : "this expression";
      ctx.addIssue(assignedExpression,
        String.format("Assign to %s a value of type \"%s\"%s or update its type hint.",
          variableMessage,
          InferredTypes.typeName(expectedType),
          inferredTypeNameMessage))
        .secondary(annotation.expression(), null);
    }
  }

  private static boolean isTypeUsedInsteadOfInstance(Expression assignedExpression, InferredType expectedType) {
    if (assignedExpression.is(Tree.Kind.NAME)) {
      Name name = (Name) assignedExpression;
      Symbol symbol = name.symbol();
      return symbol != null && symbol.is(Symbol.Kind.CLASS) &&
        !expectedType.isCompatibleWith(InferredTypes.runtimeType(TypeShed.typeShedClass("type")));
    }
    return false;
  }
}
