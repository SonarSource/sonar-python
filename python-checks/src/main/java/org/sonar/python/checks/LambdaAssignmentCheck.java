/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6661")
public class LambdaAssignmentCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, ctx -> checkAssignmentStatement(ctx, (AssignmentStatement) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, ctx -> checkAnnotatedAssignment(ctx, (AnnotatedAssignment) ctx.syntaxNode()));
  }

  private static void checkAssignmentStatement(SubscriptionContext ctx, AssignmentStatement stmt) {
    final Expression right = stmt.assignedValue();
    addIssueIfLambda(ctx, right);
  }

  private static void checkAnnotatedAssignment(SubscriptionContext ctx, AnnotatedAssignment assignment) {
    final Expression right = assignment.assignedValue();
    if (right != null) {
      addIssueIfLambda(ctx, right);
    }
  }

  private static void addIssueIfLambda(SubscriptionContext ctx, Expression assignedExpression) {
    if (assignedExpression.is(Tree.Kind.LAMBDA)) {
      final LambdaExpression lambdaExpression = (LambdaExpression) assignedExpression;
      ctx.addIssue(lambdaExpression.lambdaKeyword(), "Define function instead of this lambda assignment statement.");
    }
  }
}
