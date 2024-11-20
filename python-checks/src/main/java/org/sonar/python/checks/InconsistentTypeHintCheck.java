/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Optional;
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
    InferredType inferredType = assignedExpression.type();
    TypeAnnotation annotation = annotatedAssignment.annotation();
    InferredType expectedType = InferredTypes.fromTypeAnnotation(annotation);
    if (expectedType.mustBeOrExtend("typing.TypedDict")) {
      // Avoid FPs for TypedDict
      return;
    }
    if (!inferredType.isCompatibleWith(expectedType) || isTypeUsedInsteadOfInstance(assignedExpression, expectedType)) {
      String message = getIssueMessage(annotatedAssignment.variable(), inferredType, expectedType);

      ctx.addIssue(assignedExpression, message)
        .secondary(annotation.expression(), null);
    }
  }

  private static String getIssueMessage(Expression variable, InferredType inferredType, InferredType expectedType) {
    String expectedTypeName = InferredTypes.typeName(expectedType);
    String inferredTypeName = InferredTypes.typeName(inferredType);

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
