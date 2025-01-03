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
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S6546")
public class UnionTypeExpressionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a union type expression for this type hint.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER_TYPE_ANNOTATION, UnionTypeExpressionCheck::checkTypeAnnotation);
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_TYPE_ANNOTATION, UnionTypeExpressionCheck::checkTypeAnnotation);
    context.registerSyntaxNodeConsumer(Tree.Kind.VARIABLE_TYPE_ANNOTATION, UnionTypeExpressionCheck::checkTypeAnnotation);
  }

  private static void checkTypeAnnotation(SubscriptionContext ctx) {
    if (!supportsUnionTypeExpressions(ctx)) {
      return;
    }

    TypeAnnotation typeAnnotation = (TypeAnnotation) ctx.syntaxNode();
    Expression expression = typeAnnotation.expression();
    if (expression.is(Tree.Kind.BITWISE_OR)) {
      return;
    }

    InferredType type = InferredTypes.fromTypeAnnotation(typeAnnotation);
    String fqn = InferredTypes.fullyQualifiedTypeName(type);
    if ("typing.Union".equals(fqn)) {
      ctx.addIssue(expression, MESSAGE);
    }
  }

  private static boolean supportsUnionTypeExpressions(SubscriptionContext ctx) {
    return PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_310);
  }
}
