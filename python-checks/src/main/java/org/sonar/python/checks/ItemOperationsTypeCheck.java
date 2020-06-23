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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;
import static org.sonar.python.types.InferredTypes.typeClassLocation;

@Rule(key = "S5644")
public class ItemOperationsTypeCheck extends PythonSubscriptionCheck {

  private static final List<String> NON_DELITEM_TYPES = Arrays.asList("frozenset", "memoryview");
  private static final List<String> NON_GET_SET_ITEM_TYPES = Collections.singletonList("frozenset");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.SUBSCRIPTION, ItemOperationsTypeCheck::checkSubscription);
  }

  private static void checkSubscription(SubscriptionContext ctx) {
    SubscriptionExpression subscriptionExpression = (SubscriptionExpression) ctx.syntaxNode();
    if (isWithinTypeAnnotation(subscriptionExpression)) {
      return;
    }
    List<LocationInFile> secondaries = new ArrayList<>();
    Expression subscriptionObject = subscriptionExpression.object();
    if (isWithinDelStatement(subscriptionExpression)) {
      if (!isValidSubscription(subscriptionObject, "__delitem__", null, NON_DELITEM_TYPES, secondaries)) {
        reportIssue(subscriptionExpression, subscriptionObject, "__delitem__", ctx, secondaries);
      }
      return;
    }
    if (isWithinAssignment(subscriptionExpression)) {
      if (!isValidSubscription(subscriptionObject, "__setitem__", null, NON_GET_SET_ITEM_TYPES, secondaries)) {
        reportIssue(subscriptionExpression, subscriptionObject, "__setitem__", ctx, secondaries);
      }
      return;
    }
    if (!isValidSubscription(subscriptionObject, "__getitem__", "__class_getitem__", NON_GET_SET_ITEM_TYPES, secondaries)) {
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

  private static boolean isValidSubscription(Expression subscriptionObject, String requiredMethod, @Nullable String classRequiredMethod,
                                             List<String> invalidTypes, List<LocationInFile> secondaries) {

    if (subscriptionObject.is(Tree.Kind.GENERATOR_EXPR)) {
      return false;
    }
    if (subscriptionObject.is(Tree.Kind.CALL_EXPR)) {
      Symbol subscriptionCalleeSymbol = ((CallExpression) subscriptionObject).calleeSymbol();
      if (subscriptionCalleeSymbol != null && subscriptionCalleeSymbol.is(FUNCTION) && ((FunctionSymbol) subscriptionCalleeSymbol).isAsynchronous()) {
        secondaries.add(((FunctionSymbol) subscriptionCalleeSymbol).definitionLocation());
        return false;
      }
    }
    if (subscriptionObject instanceof HasSymbol) {
      Symbol symbol = ((HasSymbol) subscriptionObject).symbol();
      if (symbol == null) {
        return true;
      }
      if (symbol.is(FUNCTION, CLASS)) {
        secondaries.add(symbol.is(FUNCTION) ? ((FunctionSymbol) symbol).definitionLocation() : ((ClassSymbol) symbol).definitionLocation());
        return canHaveMethod(symbol, requiredMethod, classRequiredMethod);
      }
    }
    InferredType type = subscriptionObject.type();
    secondaries.add(typeClassLocation(type));
    return type.canHaveMember(requiredMethod) && invalidTypes.stream().noneMatch(type::canOnlyBe);
  }

  private static boolean canHaveMethod(Symbol symbol, String requiredMethod, @Nullable String classRequiredMethod) {
    if (symbol.is(FUNCTION)) {
      // Avoid FPs for properties
      return ((FunctionSymbol) symbol).hasDecorators();
    }
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    return classSymbol.canHaveMember(requiredMethod)
      || (classRequiredMethod != null && classSymbol.canHaveMember(classRequiredMethod));
  }

  private static void reportIssue(SubscriptionExpression subscriptionExpression, Expression subscriptionObject,
                                  String missingMethod, SubscriptionContext ctx, List<LocationInFile> secondaries) {

    String name = nameFromExpression(subscriptionObject);
    PreciseIssue preciseIssue;
    if (name != null) {
      preciseIssue = ctx.addIssue(subscriptionExpression, String.format("Fix this code; \"%s\" does not have a \"%s\" method.", name, missingMethod));
    } else {
      preciseIssue = ctx.addIssue(subscriptionObject, String.format("Fix this code; this expression does not have a \"%s\" method.", missingMethod));
    }
    secondaries.stream().filter(Objects::nonNull).forEach(locationInFile -> preciseIssue.secondary(locationInFile, null));
  }

  private static String nameFromExpression(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return ((Name) expression).name();
    }
    return null;
  }
}
