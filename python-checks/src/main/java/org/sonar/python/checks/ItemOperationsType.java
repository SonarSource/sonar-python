/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

public abstract class ItemOperationsType extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.SUBSCRIPTION, this::checkSubscription);
  }

  private void checkSubscription(SubscriptionContext ctx) {
    SubscriptionExpression subscriptionExpression = (SubscriptionExpression) ctx.syntaxNode();
    if (isWithinTypeAnnotation(subscriptionExpression)) {
      return;
    }
    List<LocationInFile> secondaries = new ArrayList<>();
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
                                  String missingMethod, SubscriptionContext ctx, List<LocationInFile> secondaries) {

    String name = nameFromExpression(subscriptionObject);
    PreciseIssue preciseIssue = ctx.addIssue(name != null ? subscriptionExpression : subscriptionObject, message(name, missingMethod));
    secondaries.stream().filter(Objects::nonNull).forEach(locationInFile -> preciseIssue.secondary(locationInFile, null));
  }

  private static String nameFromExpression(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return ((Name) expression).name();
    }
    return null;
  }

  public abstract boolean isValidSubscription(Expression subscriptionObject, String requiredMethod, @Nullable String classRequiredMethod, List<LocationInFile> secondaries);
  public abstract String message(@Nullable String name, String missingMethod);
}
