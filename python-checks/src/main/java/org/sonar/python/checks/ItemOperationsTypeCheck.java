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

import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;
import static org.sonar.python.types.InferredTypes.typeClassLocation;

@Rule(key = "S5644")
public class ItemOperationsTypeCheck extends ItemOperationsType {

  @Override
  public boolean isValidSubscription(Expression subscriptionObject, String requiredMethod, @Nullable String classRequiredMethod,
    Map<LocationInFile, String> secondaries) {

    if (subscriptionObject.is(Tree.Kind.GENERATOR_EXPR)) {
      return false;
    }
    if (isInvalidSubscriptionCallExpr(subscriptionObject, secondaries)) {
      return false;
    }

    var symbolOptional = TreeUtils.getSymbolFromTree(subscriptionObject);

    if (symbolOptional.isPresent()) {
      var symbol = symbolOptional.get();
      if (isTypingOrCollectionsSymbol(symbol)) {
        return true;
      }
      if (symbol.is(FUNCTION, CLASS)) {
        return isValidSubscriptionSymbol(symbol, subscriptionObject, secondaries, requiredMethod, classRequiredMethod);
      }
    }

    InferredType type = subscriptionObject.type();
    String typeName = InferredTypes.typeName(type);
    String secondaryMessage = typeName != null ? String.format(SECONDARY_MESSAGE, typeName) : DEFAULT_SECONDARY_MESSAGE;
    secondaries.put(typeClassLocation(type), secondaryMessage);
    return type.canHaveMember(requiredMethod);
  }

  private static boolean isValidSubscriptionSymbol(Symbol symbol, Expression subscriptionObject, Map<LocationInFile, String> secondaries, String requiredMethod,
    @Nullable String classRequiredMethod) {
    LocationInFile locationInFile = symbol.is(FUNCTION) ? ((FunctionSymbol) symbol).definitionLocation() : ((ClassSymbol) symbol).definitionLocation();
    secondaries.put(locationInFile, SECONDARY_MESSAGE.formatted(symbol.name()));
    return isSubscriptionInClassArg(subscriptionObject) 
      || canHaveMethod(symbol, requiredMethod, classRequiredMethod) 
      || isValidGenericUsage(symbol, subscriptionObject, requiredMethod);
  }

  private static boolean isValidGenericUsage(Symbol symbol, Expression subscriptionObject, String requiredMethod) {
    return "__getitem__".equals(requiredMethod) && symbol.is(CLASS) && !areSomeSubscriptsSuspicious(subscriptionObject);
  }

  private static boolean areSomeSubscriptsSuspicious(Expression subscriptionObject) {
    var subscriptionExprTree = TreeUtils.firstAncestorOfKind(subscriptionObject, Tree.Kind.SUBSCRIPTION);
    return subscriptionExprTree instanceof SubscriptionExpression subscriptionExpr 
      && subscriptionExpr.subscripts().expressions().stream()
        .allMatch(ItemOperationsTypeCheck::isSubscriptSuspicious);
  }

  private static boolean isSubscriptSuspicious(Expression expr) {
    // a subscript used as a generic should be a name of a class, alias, or a class as a string literal; Anything else is suspicious
    return !expr.is(Tree.Kind.NAME, Tree.Kind.STRING_LITERAL);
  }

  private static boolean isInvalidSubscriptionCallExpr(Expression expression, Map<LocationInFile, String> secondaries) {
    if (expression instanceof CallExpression callExpression
      && callExpression.calleeSymbol() instanceof FunctionSymbol functionSymbol
      && functionSymbol.isAsynchronous()) {
      secondaries.put(functionSymbol.definitionLocation(), SECONDARY_MESSAGE.formatted(functionSymbol.name()));
      return true;
    }
    return false;
  }

  private static boolean isSubscriptionInClassArg(Expression subscriptionObject) {
    var classDefOptional = Optional.ofNullable(TreeUtils.firstAncestorOfKind(subscriptionObject, Tree.Kind.CLASSDEF))
      .map(ClassDef.class::cast);

    List<Argument> classArguments = classDefOptional
      .map(ClassDef::args)
      .map(ArgList::arguments)
      .orElse(List.of());

    var onlyRegularArgumentExpressions = classArguments.stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
      .map(RegularArgument::expression);

    var subscriptionObjectStream = onlyRegularArgumentExpressions
      .flatMap(TreeUtils.toStreamInstanceOfMapper(SubscriptionExpression.class))
      .map(SubscriptionExpression::object);

    return subscriptionObjectStream.anyMatch(subscriptionObject::equals);
  }

  @Override
  public String message(@Nullable String name, String missingMethod) {
    if (name != null) {
      return String.format("Fix this code; \"%s\" does not have a \"%s\" method.", name, missingMethod);
    }
    return String.format("Fix this code; this expression does not have a \"%s\" method.", missingMethod);
  }

  private static boolean isTypingOrCollectionsSymbol(Symbol symbol) {
    String fullyQualifiedName = symbol.fullyQualifiedName();
    // avoid FP for typing symbols like 'Awaitable[None]'
    return fullyQualifiedName != null && (fullyQualifiedName.startsWith("typing") || fullyQualifiedName.startsWith("collections"));
  }

  private static boolean canHaveMethod(Symbol symbol, String requiredMethod, @Nullable String classRequiredMethod) {
    if (symbol.is(FUNCTION)) {
      // Avoid FPs for properties
      return ((FunctionSymbol) symbol).hasDecorators();
    }
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    return classSymbol.canHaveMember(requiredMethod)
      || (classRequiredMethod != null && classSymbol.canHaveMember(classRequiredMethod))
      || classSymbol.hasDecorators();
  }
}
