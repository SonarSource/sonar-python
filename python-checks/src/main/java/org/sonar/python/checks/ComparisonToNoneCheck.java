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
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeChecker;
import org.sonar.python.types.v2.TypeUtils;

@Rule(key = "S5727")
public class ComparisonToNoneCheck extends PythonSubscriptionCheck {
  private TypeCheckBuilder canBeNoneTypeCheck;
  private TypeCheckBuilder isNoneTypeCheck;
  private TypeCheckBuilder isObjectTypeCheck;
  private TypeCheckBuilder isExactTypeSource;
  private TypeChecker typeChecker;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, ctx -> {
      typeChecker = ctx.typeChecker();
      canBeNoneTypeCheck = ctx.typeChecker().typeCheckBuilder().isExactTypeSource().canBeBuiltinWithName(BuiltinTypes.NONE_TYPE);
      isNoneTypeCheck = ctx.typeChecker().typeCheckBuilder().isExactTypeSource().isBuiltinWithName(BuiltinTypes.NONE_TYPE);
      isObjectTypeCheck = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName(BuiltinTypes.OBJECT_TYPE);
      isExactTypeSource = ctx.typeChecker().typeCheckBuilder().isExactTypeSource();
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
    if (isTypeHint(left) || isTypeHint(right)) {
      return;
    }
    if (isStubProbablyWrong(left) || isStubProbablyWrong(right)) {
      return;
    }
    if (isPropertyDecoratedFunction(left) || isPropertyDecoratedFunction(right)) {
      return;
    }
    if (isNoneCheckAlwaysTrue(left, right)) {
      addIssue(ctx, comparison, operator + " comparison", "==".equals(operator));
    } else if (isNoneCheckImpossible(left, right)) {
      addIssue(ctx, comparison, operator + " comparison", "!=".equals(operator));
    }
  }

  private void checkIdentityComparison(SubscriptionContext ctx, IsExpression comparison) {
    PythonType left = comparison.leftOperand().typeV2();
    PythonType right = comparison.rightOperand().typeV2();
    if (isStubProbablyWrong(left) || isStubProbablyWrong(right)) {
      return;
    }
    if (isTypeHint(left) || isTypeHint(right)) {
      return;
    }
    if (isPropertyDecoratedFunction(left) || isPropertyDecoratedFunction(right)) {
      return;
    }
    if (isNotIdentityComparableWith(left, right) && isNoneCheckImpossible(left, right)) {
      addIssue(ctx, comparison, "identity check", comparison.notToken() != null);
    } else if (isNoneCheckAlwaysTrue(left, right)) {
      addIssue(ctx, comparison, "identity check", comparison.notToken() == null);
    }
  }

  private boolean isNoneCheckImpossible(PythonType left, PythonType right) {
    return (isNone(left) && cannotBeNone(right)) || (cannotBeNone(left) && isNone(right));
  }

  private boolean isNoneCheckAlwaysTrue(PythonType left, PythonType right) {
    return isNone(left) && isNone(right);
  }

  private boolean cannotBeNone(PythonType type) {
    return canBeNoneTypeCheck.check(TypeUtils.unwrapType(type)) == TriBool.FALSE;
  }

  private boolean isNone(PythonType type) {
    return isNoneTypeCheck.check(TypeUtils.unwrapType(type)) == TriBool.TRUE;
  }

  private boolean isNotIdentityComparableWith(PythonType left, PythonType right) {
    return typeChecker.typeCheckBuilder().isIdentityComparableWith(left).check(right) == TriBool.FALSE;
  }

  private boolean isStubProbablyWrong(PythonType type) {
    // we assume that if a stub returns an object that its type is probably not correct and the author probably meant writing `Any`
    return isObjectTypeCheck.check(type) == TriBool.TRUE;
  }

  private boolean isTypeHint(PythonType type) {
    return isExactTypeSource.check(type) == TriBool.FALSE;
  }

  private static boolean isPropertyDecoratedFunction(PythonType type) {
    //TODO SONAR-1772 actually detect decorated functions (or remove check entirely if type is correctly resolved)
    return type instanceof FunctionType;
  }

  private static void addIssue(SubscriptionContext ctx, Tree tree, String comparisonKind, boolean result) {
    String resultAsString = result ? "True" : "False";
    ctx.addIssue(tree, String.format("Remove this %s; it will always be %s.", comparisonKind, resultAsString));
  }
}
