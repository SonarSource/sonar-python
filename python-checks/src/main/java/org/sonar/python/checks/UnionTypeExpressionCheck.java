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
    PythonVersionUtils.Version required = PythonVersionUtils.Version.V_310;

    // All versions must be greater than or equal to the required version.
    return ctx.sourcePythonVersions().stream()
      .allMatch(version -> version.compare(required.major(), required.minor()) >= 0);
  }
}
