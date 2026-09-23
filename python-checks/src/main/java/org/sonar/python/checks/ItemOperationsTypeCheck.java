/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.types.v2.FullyQualifiedNameHelper;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;
import static org.sonar.python.types.InferredTypes.typeClassLocation;

@Rule(key = "S5644")
public class ItemOperationsTypeCheck extends ItemOperationsType {

  private static final Set<String> TYPING_LITERAL = Set.of("typing.Literal", "typing_extensions.Literal");
  private static final Set<String> TYPING_ANNOTATED = Set.of("typing.Annotated", "typing_extensions.Annotated");

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
    return "__getitem__".equals(requiredMethod)
      && symbol.is(CLASS)
      && subscriptionObject.typeV2() instanceof ClassType classType
      && classType.isGeneric()
      && !areSomeSubscriptsSuspicious(subscriptionObject);
  }

  private static boolean areSomeSubscriptsSuspicious(Expression subscriptionObject) {
    var subscriptionExprTree = TreeUtils.firstAncestorOfKind(subscriptionObject, Tree.Kind.SUBSCRIPTION);
    return subscriptionExprTree instanceof SubscriptionExpression subscriptionExpr 
      && subscriptionExpr.subscripts().expressions().stream()
        .anyMatch(ItemOperationsTypeCheck::isSubscriptSuspicious);
  }

  private static boolean isSubscriptSuspicious(Expression expr) {
    if (expr.is(Tree.Kind.NAME, Tree.Kind.STRING_LITERAL, Tree.Kind.NONE, Tree.Kind.ELLIPSIS)) {
      return false;
    }
    if (expr instanceof ParenthesizedExpression parenthesizedExpression) {
      return isSubscriptSuspicious(parenthesizedExpression.expression());
    }
    if (expr instanceof BinaryExpression binaryExpression && expr.is(Tree.Kind.BITWISE_OR)) {
      return isSubscriptSuspicious(binaryExpression.leftOperand()) || isSubscriptSuspicious(binaryExpression.rightOperand());
    }
    if (expr instanceof UnpackingExpression unpackingExpression) {
      return isSubscriptSuspicious(unpackingExpression.expression());
    }
    if (expr instanceof ListLiteral listLiteral) {
      return listLiteral.elements().expressions().stream().anyMatch(ItemOperationsTypeCheck::isSubscriptSuspicious);
    }
    if (expr instanceof Tuple tuple) {
      return tuple.elements().stream().anyMatch(ItemOperationsTypeCheck::isSubscriptSuspicious);
    }
    if (expr instanceof QualifiedExpression qualifiedExpression) {
      return isSubscriptSuspicious(qualifiedExpression.qualifier());
    }
    if (expr instanceof SubscriptionExpression subscriptionExpression) {
      return isSubscriptionSuspicious(subscriptionExpression);
    }
    return true;
  }

  private static boolean isSubscriptionSuspicious(SubscriptionExpression subscriptionExpression) {
    List<Expression> arguments = subscriptionExpression.subscripts().expressions();
    if (hasType(subscriptionExpression.object(), TYPING_LITERAL)) {
      return false;
    }
    if (hasType(subscriptionExpression.object(), TYPING_ANNOTATED)) {
      return arguments.isEmpty() || isSubscriptSuspicious(arguments.get(0));
    }
    return isSubscriptSuspicious(subscriptionExpression.object())
      || arguments.stream().anyMatch(ItemOperationsTypeCheck::isSubscriptSuspicious);
  }

  private static boolean hasType(Expression expression, Set<String> fullyQualifiedNames) {
    return FullyQualifiedNameHelper.getFullyQualifiedName(expression.typeV2())
      .filter(fullyQualifiedNames::contains)
      .isPresent();
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
