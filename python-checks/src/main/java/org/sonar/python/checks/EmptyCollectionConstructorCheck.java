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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7498")
public class EmptyCollectionConstructorCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this constructor call with a literal.";
  private static final List<String> COLLECTION_CONSTRUCTORS = Arrays.asList("dict", "list", "tuple");

  private List<TypeCheckBuilder> collectionConstructorTypeCheckers = null;
  private TypeCheckBuilder dictChecker = null;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      dictChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("dict");

      collectionConstructorTypeCheckers = new ArrayList<>();
      for (String constructor : COLLECTION_CONSTRUCTORS) {
        collectionConstructorTypeCheckers.add(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(constructor));
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      if (isUnnecessaryCollectionConstructor(callExpression)) {
        ctx.addIssue(callExpression.callee(), MESSAGE);
      }
    });
  }

  private boolean isUnnecessaryCollectionConstructor(CallExpression callExpression) {
    return (isCollectionConstructor(callExpression.callee()) && isEmptyCall(callExpression))
      || isDictConstructorWithOnlyMappings(callExpression);
  }

  private boolean isCollectionConstructor(Expression calleeExpression) {
    var type = calleeExpression.typeV2();
    return collectionConstructorTypeCheckers.stream().map(checker -> checker.check(type)).anyMatch(TriBool.TRUE::equals);
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
}
