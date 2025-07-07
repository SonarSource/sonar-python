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

import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.checks.utils.IsComprehensionTransformedChecker;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7496")
public class CollectionCreationWrappedInConstructorCheck extends PythonSubscriptionCheck {

  private static final String TUPLE = "tuple";
  private static final String SET = "set";
  private static final String LIST = "list";
  private static final String DICT = "dict";

  private static final Set<String> COLLECTION_TYPES = Set.of(LIST, SET, DICT, TUPLE);

  private static final Set<Kind> COLLECTION_KINDS = EnumSet.of(
    Kind.LIST_LITERAL, Kind.SET_LITERAL, Kind.DICTIONARY_LITERAL, Kind.TUPLE,
    Kind.LIST_COMPREHENSION, Kind.SET_COMPREHENSION, Kind.DICT_COMPREHENSION
  );

  private static final Map<String, List<Kind>> SAME_INNER_TYPES_MAP = Map.of(
    LIST, List.of(Kind.LIST_LITERAL, Kind.LIST_COMPREHENSION),
    SET, List.of(Kind.SET_LITERAL, Kind.SET_COMPREHENSION),
    DICT, List.of(Kind.DICTIONARY_LITERAL, Kind.DICT_COMPREHENSION),
    TUPLE, List.of(Kind.TUPLE)
  );

  private final TypeCheckMap<String> collectionTypeCheckerMap = new TypeCheckMap<>();
  private IsComprehensionTransformedChecker isComprehensionTransformedChecker = null;

  @Override
  public void initialize(Context context) {
    isComprehensionTransformedChecker = new IsComprehensionTransformedChecker(context);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeTypeCheckers);
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, this::checkCalls);
  }

  private void initializeTypeCheckers(SubscriptionContext ctx) {
    for (var collectionType : COLLECTION_TYPES) {
      TypeCheckBuilder typeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(collectionType);
      collectionTypeCheckerMap.put(typeChecker, collectionType);
    }
  }

  private void checkCalls(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Expression callee = callExpression.callee();
    RegularArgument regularArgument = getSingleRegularArgument(callExpression);
    if (regularArgument == null) {
      return;
    }

    if (regularArgument.expression() instanceof ComprehensionExpression comprehensionExpression
      && !isComprehensionTransformedChecker.isGeneratorTransformingData(comprehensionExpression, callee.typeV2())) {
      return;
    }

    var collectionType = collectionTypeCheckerMap.getForType(callee.typeV2());
    if (collectionType == null || callee instanceof SubscriptionExpression) {
      return;
    }

    if (isSameInnerType(collectionType, regularArgument)) {
      ctx.addIssue(callExpression.callee(), getSameInnerTypeCollectionMessage(collectionType));
    } else if (isSensitiveDifferentLiteralInnerType(collectionType, regularArgument)) {
      ctx.addIssue(callExpression.callee(), getDifferentInnerTypeCollectionMessage(collectionType));
    } else if (collectionType.equals(TUPLE)) {
      handleTupleConstructorSpecialCases(ctx, regularArgument);
    }
  }

  private static RegularArgument getSingleRegularArgument(CallExpression callExpression) {
    if (callExpression.arguments().size() != 1) {
      return null;
    }
    Argument argument = callExpression.arguments().get(0);
    if (!(argument instanceof RegularArgument regularArgument)) {
      return null;
    }
    return regularArgument;
  }

  private static boolean isSameInnerType(String collectionType, RegularArgument regularArgument) {
    return SAME_INNER_TYPES_MAP.get(collectionType)
      .stream()
      .anyMatch(kind -> regularArgument.expression().is(kind));
  }

  private static boolean isSensitiveDifferentLiteralInnerType(String collectionType, RegularArgument regularArgument) {
    if (!isCollectionKind(regularArgument)) {
      return false;
    }

    if (isSameInnerType(collectionType, regularArgument)) {
      return false;
    }

    // It is common and valid to check for duplicates using set comprehension before creating another collection
    if (regularArgument.expression().is(Kind.SET_COMPREHENSION)) {
      return false;
    }

    // When tuple are created from comprehensions, the issue is handled in handleTupleConstructorSpecialCases as it requires a special message
    EnumSet<Kind> comprehensions = EnumSet.of(Kind.LIST_COMPREHENSION, Kind.SET_COMPREHENSION, Kind.DICT_COMPREHENSION);
    return !collectionType.equals(TUPLE) || !comprehensions.contains(regularArgument.expression().getKind());
  }

  private static String getSameInnerTypeCollectionMessage(String type) {
    return "Remove the redundant " + type + " constructor call.";
  }

  private static String getDifferentInnerTypeCollectionMessage(String type) {
    return "Replace this " + type + " constructor call by a " + type + " literal.";
  }

  private static void handleTupleConstructorSpecialCases(SubscriptionContext ctx, RegularArgument regularArgument) {
    if (regularArgument.expression().is(Kind.LIST_COMPREHENSION)) {
      ctx.addIssue(regularArgument, "Replace this list comprehension by a generator.");
    } else if (regularArgument.expression().is(Kind.SET_COMPREHENSION)) {
      ctx.addIssue(regularArgument, "Replace this set comprehension by a generator.");
    } else if (regularArgument.expression().is(Kind.DICT_COMPREHENSION)) {
      ctx.addIssue(regularArgument, "Replace this dict comprehension by a generator.");
    }
  }

  private static boolean isCollectionKind(RegularArgument regularArgument) {
    return COLLECTION_KINDS.contains(regularArgument.expression().getKind());
  }
}
