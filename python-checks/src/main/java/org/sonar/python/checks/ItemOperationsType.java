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

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.tree.TreeUtils.getSymbolFromTree;
import static org.sonar.python.tree.TreeUtils.nameFromExpression;

public abstract class ItemOperationsType extends PythonSubscriptionCheck {

  static final String SECONDARY_MESSAGE = "Definition of \"%s\".";
  static final String DEFAULT_SECONDARY_MESSAGE = "Type definition.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.SUBSCRIPTION, this::checkSubscription);
  }

  private void checkSubscription(SubscriptionContext ctx) {
    SubscriptionExpression subscriptionExpression = (SubscriptionExpression) ctx.syntaxNode();
    if (isWithinTypeAnnotation(subscriptionExpression) || isTypeSubscription(subscriptionExpression.object())) {
      return;
    }
    Map<LocationInFile, String> secondaries = new HashMap<>();
    Expression subscriptionObject = subscriptionExpression.object();
    if (isWithinDelStatement(subscriptionExpression)) {
      if (!isValidSubscription(subscriptionObject, "__delitem__", null, secondaries)) {
        reportIssue(subscriptionExpression, subscriptionObject, "__delitem__", ctx, secondaries);
      }
      return;
    }
    if (isWithinAssignment(subscriptionExpression)) {
      if (!isValidSubscription(subscriptionObject, "__setitem__", null, secondaries)) {
        reportIssue(subscriptionExpression, subscriptionObject, "__setitem__", ctx, secondaries);
      }
      return;
    }
    if (!isValidSubscription(subscriptionObject, "__getitem__", "__class_getitem__", secondaries)) {
      reportIssue(subscriptionExpression, subscriptionObject, "__getitem__", ctx, secondaries);
    }
  }

  // Avoid FPs on alias with type generics like `alias = type[Foo]`
  private static boolean isTypeSubscription(Expression subscriptionObject) {
    return getSymbolFromTree(subscriptionObject).filter(symbol -> "type".equals(symbol.fullyQualifiedName())).isPresent();
  }

  private static boolean isWithinTypeAnnotation(SubscriptionExpression subscriptionExpression) {
    return TreeUtils.firstAncestor(subscriptionExpression,
      t -> t.is(Tree.Kind.PARAMETER_TYPE_ANNOTATION, Tree.Kind.RETURN_TYPE_ANNOTATION, Tree.Kind.VARIABLE_TYPE_ANNOTATION)) != null;
  }

  private static boolean isWithinDelStatement(SubscriptionExpression subscriptionExpression) {
    return TreeUtils.firstAncestor(subscriptionExpression,
      t -> t.is(Tree.Kind.DEL_STMT) && ((DelStatement) t).expressions().stream()
        .anyMatch(e -> e.equals(subscriptionExpression))) != null;
  }

  private static boolean isWithinAssignment(SubscriptionExpression subscriptionExpression) {
    return TreeUtils.firstAncestor(subscriptionExpression,
      t -> t.is(Tree.Kind.ASSIGNMENT_STMT) && ((AssignmentStatement) t).lhsExpressions().stream().flatMap(lhs -> lhs.expressions().stream())
        .anyMatch(e -> e.equals(subscriptionExpression))) != null;
  }

  private void reportIssue(SubscriptionExpression subscriptionExpression, Expression subscriptionObject,
                           String missingMethod, SubscriptionContext ctx, Map<LocationInFile, String> secondaries) {

    String name = nameFromExpression(subscriptionObject);
    PreciseIssue preciseIssue = ctx.addIssue(name != null ? subscriptionExpression : subscriptionObject, message(name, missingMethod));
    secondaries.keySet().stream().filter(Objects::nonNull).forEach(location -> preciseIssue.secondary(location, secondaries.get(location)));
  }

  public abstract boolean isValidSubscription(Expression subscriptionObject, String requiredMethod, @Nullable String classRequiredMethod, Map<LocationInFile, String> secondaries);

  public abstract String message(@Nullable String name, String missingMethod);
}
