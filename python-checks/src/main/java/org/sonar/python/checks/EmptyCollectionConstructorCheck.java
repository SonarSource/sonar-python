/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.Map;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7498")
public class EmptyCollectionConstructorCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this constructor call with a literal.";
  private static final Map<String, String> COLLECTION_CONSTRUCTORS = Map.ofEntries(
    Map.entry("list", "[]"),
    Map.entry("tuple", "()"),
    Map.entry("dict", "{}")
  );

  private TypeCheckMap<String> collectionConstructorTypeCheckers = null;
  private TypeCheckBuilder dictChecker = null;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      dictChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("dict");

      collectionConstructorTypeCheckers = new TypeCheckMap<>();
      for (var constructorEntry : COLLECTION_CONSTRUCTORS.entrySet()) {
        TypeCheckBuilder constructorTypeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(constructorEntry.getKey());
        collectionConstructorTypeCheckers.put(constructorTypeChecker, constructorEntry.getValue());
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      if (isUnnecessaryCollectionConstructor(callExpression)) {
        var issue = ctx.addIssue(callExpression.callee(), MESSAGE);
        createQuickFix(callExpression).ifPresent(issue::addQuickFix);
      }
    });
  }

  private boolean isUnnecessaryCollectionConstructor(CallExpression callExpression) {
    return (isCollectionConstructor(callExpression.callee()) && isEmptyCall(callExpression))
      || isDictConstructorWithOnlyMappings(callExpression);
  }

  private boolean isCollectionConstructor(Expression calleeExpression) {
    var type = calleeExpression.typeV2();
    return collectionConstructorTypeCheckers.getOptionalForType(type).isPresent() && !(calleeExpression instanceof SubscriptionExpression);
  }

  private static boolean isEmptyCall(CallExpression callExpression) {
    return callExpression.arguments().isEmpty();
  }

  private boolean isDictConstructorWithOnlyMappings(CallExpression callExpression) {
    return isDictConstructor(callExpression) && hasOnlyKeywordArguments(callExpression);
  }

  private boolean isDictConstructor(CallExpression callExpression) {
    return dictChecker.check(callExpression.callee().typeV2()) == TriBool.TRUE;
  }

  private static boolean hasOnlyKeywordArguments(CallExpression callExpression) {
    return callExpression.arguments().stream().allMatch(EmptyCollectionConstructorCheck::isKeywordArg);
  }

  private static boolean isKeywordArg(Argument arg) {
    return arg instanceof RegularArgument regularArg && regularArg.keywordArgument() != null;
  }

  private Optional<PythonQuickFix> createQuickFix(CallExpression callExpression) {
    return Optional.of(callExpression)
      .filter(EmptyCollectionConstructorCheck::isEmptyCall)
      .flatMap(callExpr -> collectionConstructorTypeCheckers.getOptionalForType(callExpression.callee().typeV2()))
      .map(replacementStr -> PythonQuickFix.newQuickFix("Replace with literal", TextEditUtils.replace(callExpression, replacementStr)));
  }
}
