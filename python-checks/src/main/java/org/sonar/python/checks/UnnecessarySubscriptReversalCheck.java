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

import java.util.Optional;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.SliceItem;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7511")
public class UnnecessarySubscriptReversalCheck extends PythonSubscriptionCheck {
  private TypeCheckBuilder isReversedTypeCheck;
  private TypeCheckBuilder isSortedTypeCheck;
  private TypeCheckBuilder isSetTypeCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initChecks(SubscriptionContext ctx) {
    isReversedTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("reversed");
    isSortedTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("sorted");
    isSetTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("set");
  }

  private void check(SubscriptionContext ctx) {
    TreeUtils.toOptionalInstanceOf(CallExpression.class, ctx.syntaxNode())
      .filter(this::isSensitiveCall)
      .flatMap(callExpression -> getArgumentSubscriptReversal(callExpression)
        .or(() -> getOuterSubscriptReversal(callExpression)))
      .ifPresent(redundantSubscriptReversal -> ctx.addIssue(redundantSubscriptReversal, "Remove this redundant subscript reversal"));
  }

  private static Optional<Tree> getOuterSubscriptReversal(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(Tree::parent)
      .filter(UnnecessarySubscriptReversalCheck::isSubscriptReversal);
  }

  private static Optional<Tree> getArgumentSubscriptReversal(CallExpression callExpression) {
    return TreeUtils.nthArgumentOrKeywordOptional(0, "", callExpression.arguments())
      .map(RegularArgument::expression)
      .filter(arg -> isSubscriptReversal(arg) || isAssignedToSubscriptReversal(arg))
      .map(Tree.class::cast);
  }

  private static boolean isAssignedToSubscriptReversal(Expression argumentExpression) {
    return argumentExpression instanceof Name name
           && getUsageCount(name) == 2
           && isSubscriptReversal(Expressions.singleAssignedValue(name));
  }

  private boolean isSensitiveCall(CallExpression callExpression) {
    return Stream.of(isReversedTypeCheck, isSortedTypeCheck, isSetTypeCheck)
      .map(check -> check.check(callExpression.callee().typeV2()))
      .anyMatch(TriBool.TRUE::equals);
  }

  private static boolean isSubscriptReversal(@Nullable Tree expression) {
    return expression instanceof SliceExpression sliceExpression
           && sliceExpression.sliceList().slices().size() == 1
           && sliceExpression.sliceList().slices().get(0) instanceof SliceItem sliceItem
           && sliceItem.lowerBound() == null
           && sliceItem.upperBound() == null
           && sliceItem.stride() instanceof UnaryExpression unaryExpression
           && unaryExpression.is(Tree.Kind.UNARY_MINUS)
           && unaryExpression.expression() instanceof NumericLiteral numericLiteral
           && numericLiteral.valueAsLong() == 1;
  }

  private static int getUsageCount(Name name) {
    var symbol = name.symbolV2();
    if (symbol == null) return 0;
    return symbol.usages().size();
  }
}
