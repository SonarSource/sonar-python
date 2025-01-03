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

import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.SliceItem;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.SLICE_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.SLICE_ITEM;
import static org.sonar.plugins.python.api.tree.Tree.Kind.SUBSCRIPTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

@Rule(key = "S6663")
public class IndexMethodCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Make sure this object defines an `__index__` method.";
  private static final Set<String> ALLOWED_TYPES = Set.of(NONE_TYPE, "slice");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(SUBSCRIPTION, ctx -> checkSubscription(ctx, ((SubscriptionExpression) ctx.syntaxNode())));
    context.registerSyntaxNodeConsumer(SLICE_ITEM, ctx -> checkSliceItem(ctx, ((SliceItem) ctx.syntaxNode())));
  }

  private static void checkSliceItem(SubscriptionContext ctx, SliceItem sliceItem) {
    SliceExpression sliceExpression = ((SliceExpression) TreeUtils.firstAncestorOfKind(sliceItem, SLICE_EXPR));
    // defensive programming: slice items should always have a SliceExpression parent
    if (sliceExpression != null && !sliceExpression.object().type().mustBeOrExtend("typing.Sequence")) return;
    if (!isValidIndex(sliceItem.lowerBound())) {
      ctx.addIssue(sliceItem.lowerBound(), MESSAGE);
    }
    if (!isValidIndex(sliceItem.upperBound())) {
      ctx.addIssue(sliceItem.upperBound(), MESSAGE);
    }
    if (!isValidIndex(sliceItem.stride())) {
      ctx.addIssue(sliceItem.stride(), MESSAGE);
    }
  }

  private static void checkSubscription(SubscriptionContext ctx, SubscriptionExpression subscriptionExpression) {
    if (!subscriptionExpression.object().type().mustBeOrExtend("typing.Sequence")) return;
    var expressionList = subscriptionExpression.subscripts();
    if (!expressionList.commas().isEmpty()) {
      // if contains at least a comma, its type is going to be tuple, hence not a valid index
      ctx.addIssue(expressionList, MESSAGE);
    }
    Expression expressionIndex = expressionList.expressions().get(0);
    if (!isValidIndex(expressionIndex)) {
      ctx.addIssue(expressionIndex, MESSAGE);
    }
  }

  private static boolean isValidIndex(@Nullable Expression expressionIndex) {
    if (expressionIndex == null || ALLOWED_TYPES.stream().anyMatch(t -> expressionIndex.type().canOnlyBe(t))) return true;
    return expressionIndex.type().canHaveMember("__index__");
  }
}
