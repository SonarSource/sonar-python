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
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeChecker;

@Rule(key = "S5727")
public class ComparisonToNoneCheck extends PythonSubscriptionCheck {
  private TypeCheckBuilder isNoneTypeCheck;
  private TypeCheckBuilder isObjectTypeCheck;
  private TypeChecker typeChecker;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, ctx -> {
      typeChecker = ctx.typeChecker();
      isNoneTypeCheck = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName(BuiltinTypes.NONE_TYPE);
      isObjectTypeCheck = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName(BuiltinTypes.OBJECT_TYPE);
    });

    context.registerSyntaxNodeConsumer(Kind.IS, ctx -> checkIdentityComparison(ctx, (IsExpression) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Kind.COMPARISON, ctx -> checkEqualityComparison(ctx, (BinaryExpression) ctx.syntaxNode()));
  }

  private void checkEqualityComparison(SubscriptionContext ctx, BinaryExpression comparison) {
    String operator = comparison.operator().value();
    if (!"==".equals(operator) && !"!=".equals(operator)) {
      return;
    }
    PythonType left = comparison.leftOperand().typeV2();
    PythonType right = comparison.rightOperand().typeV2();
    if (isNone(left) && isNone(right)) {
      addIssue(ctx, comparison, operator + " comparison", "==".equals(operator));
    } else if ((isNone(left) && cannotBeNone(right)) || (cannotBeNone(left) && isNone(right))) {
      addIssue(ctx, comparison, operator + " comparison", "!=".equals(operator));
    }
  }

  private void checkIdentityComparison(SubscriptionContext ctx, IsExpression comparison) {
    PythonType left = comparison.leftOperand().typeV2();
    PythonType right = comparison.rightOperand().typeV2();
    // `isObject` Removes FP when the return type of a function is an object
    if (isNotIdentityComparableWith(left, right) && (isNone(left) || isNone(right)) && !(isObject(left) || isObject(right))) {
      addIssue(ctx, comparison, "identity check", comparison.notToken() != null);
    } else if (isNone(left) && isNone(right)) {
      addIssue(ctx, comparison, "identity check", comparison.notToken() == null);
    }
  }

  private boolean isNone(PythonType left) {
    return isNoneTypeCheck.check(left) == TriBool.TRUE;
  }

  private boolean isNotIdentityComparableWith(PythonType left, PythonType right) {
    return typeChecker.typeCheckBuilder().isIdentityComparableWith(left).check(right) == TriBool.FALSE;
  }

  private boolean cannotBeNone(PythonType type) {
    return isNoneTypeCheck.check(type) == TriBool.FALSE && isObjectTypeCheck.check(type) == TriBool.FALSE;
  }

  private boolean isObject(PythonType type) {
    return isObjectTypeCheck.check(type) == TriBool.TRUE;
  }

  private static void addIssue(SubscriptionContext ctx, Tree tree, String comparisonKind, boolean result) {
    String resultAsString = result ? "True" : "False";
    ctx.addIssue(tree, String.format("Remove this %s; it will always be %s.", comparisonKind, resultAsString));
  }
}
