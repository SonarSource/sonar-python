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

import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.IsComprehensionTransformedChecker;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7494")
public class RewriteCollectionConstructorAsComprehensionCheck extends PythonSubscriptionCheck {

  private static final Map<String, String> COLLECTION_MESSAGES = Map.of(
    "list", "Replace list constructor call with a list comprehension.",
    "set", "Replace set constructor call with a set comprehension.",
    "dict", "Replace dict constructor call with a dictionary comprehension."
  );

  private TypeCheckMap<String> collectionTypeCheckerMap = null;
  private TypeCheckBuilder tupleTypeCheck = null;
  private TypeCheckBuilder dictTypeCheck = null;
  private IsComprehensionTransformedChecker isComprehensionTransformedChecker = null;


  @Override
  public void initialize(Context context) {
    isComprehensionTransformedChecker = new IsComprehensionTransformedChecker(context);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initTypeChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  private void initTypeChecks(SubscriptionContext ctx) {
    collectionTypeCheckerMap = new TypeCheckMap<>();
    for (var collectionEntry : COLLECTION_MESSAGES.entrySet()) {
      TypeCheckBuilder typeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(collectionEntry.getKey());
      collectionTypeCheckerMap.put(typeChecker, collectionEntry.getValue());
    }
    tupleTypeCheck = ctx.typeChecker().typeCheckBuilder().isInstanceOf("tuple");
    dictTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("dict");
  }

  private void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    List<Argument> arguments = callExpression.arguments();

    String message = getMessageForConstructor(callExpression).orElse(null);
    if (message == null) {
      return;
    }

    ComprehensionExpression generator = getSingleGeneratorArg(arguments).orElse(null);
    if (generator == null || isDictCallWithTuplesResultGenerator(callExpression, generator)) {
      return;
    }

    if (isComprehensionTransformedChecker.isGeneratorTransformingData(generator, callExpression.callee().typeV2())) {
      ctx.addIssue(callExpression.callee(), message);
    }
  }

  private boolean isDictCallWithTuplesResultGenerator(CallExpression callExpression, ComprehensionExpression generatorArgument) {
    return dictTypeCheck.check(callExpression.callee().typeV2()) == TriBool.TRUE && isCallReturningTuple(generatorArgument.resultExpression());
  }

  private boolean isCallReturningTuple(Expression expression) {
    return expression instanceof CallExpression callExpression && tupleTypeCheck.check(callExpression.typeV2()) == TriBool.TRUE;
  }

  private Optional<String> getMessageForConstructor(CallExpression callExpression) {
    return collectionTypeCheckerMap.getOptionalForType(callExpression.callee().typeV2());
  }

  private static Optional<ComprehensionExpression> getSingleGeneratorArg(List<Argument> arguments) {
    if (arguments.size() != 1) {
      return Optional.empty();
    }

    if (!(arguments.get(0) instanceof RegularArgument regularArg)) {
      return Optional.empty();
    }

    Expression argument = regularArg.expression();
    if (!argument.is(Tree.Kind.GENERATOR_EXPR)) {
      return Optional.empty();
    }

    return Optional.of((ComprehensionExpression) argument);
  }
}
