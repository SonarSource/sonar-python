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
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;

import static org.sonar.python.checks.utils.CheckUtils.isNone;

@Rule(key = "S5727")
public class ComparisonToNoneCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.IS, ctx -> checkIdentityComparison(ctx, (IsExpression) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Kind.COMPARISON, ctx -> checkEqualityComparison(ctx, (BinaryExpression) ctx.syntaxNode()));
  }

  private static void checkEqualityComparison(SubscriptionContext ctx, BinaryExpression comparison) {
    String operator = comparison.operator().value();
    if (!"==".equals(operator) && !"!=".equals(operator)) {
      return;
    }
    InferredType left = comparison.leftOperand().type();
    InferredType right = comparison.rightOperand().type();
    if (isNone(left) && isNone(right)) {
      addIssue(ctx, comparison, operator + " comparison", "==".equals(operator));
    } else if ((isNone(left) && cannotBeNone(right)) || (cannotBeNone(left) && isNone(right))) {
      addIssue(ctx, comparison, operator + " comparison", "!=".equals(operator));
    }
  }

  private static void checkIdentityComparison(SubscriptionContext ctx, IsExpression comparison) {
    InferredType left = comparison.leftOperand().type();
    InferredType right = comparison.rightOperand().type();
    // `isObject` Removes FP when the return type of a function is an object
    if (!left.isIdentityComparableWith(right) && (isNone(left) || isNone(right)) && !(isObject(left) || isObject(right))) {
      addIssue(ctx, comparison, "identity check", comparison.notToken() != null);
    } else if (isNone(left) && isNone(right)) {
      addIssue(ctx, comparison, "identity check", comparison.notToken() == null);
    }
  }

  private static void addIssue(SubscriptionContext ctx, Tree tree, String comparisonKind, boolean result) {
    String resultAsString = result ? "True" : "False";
    ctx.addIssue(tree, String.format("Remove this %s; it will always be %s.", comparisonKind, resultAsString));
  }

  private static boolean cannotBeNone(InferredType type) {
    return !type.canBeOrExtend(BuiltinTypes.NONE_TYPE) && !isObject(type);
  }

  private static boolean isObject(InferredType type) {
    return type.canOnlyBe(BuiltinTypes.OBJECT_TYPE);
  }

}
