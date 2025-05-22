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
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionClause;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.semantic.v2.SymbolV2;
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
  private TypeCheckBuilder dictTypeChecker = null;

  @Override
  public void initialize(Context context) {
    collectionTypeCheckerMap = new TypeCheckMap<>();
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      for (var collectionEntry : COLLECTION_MESSAGES.entrySet()) {
        TypeCheckBuilder typeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(collectionEntry.getKey());
        collectionTypeCheckerMap.put(typeChecker, collectionEntry.getValue());

        if ("dict".equals(collectionEntry.getKey())) {
          dictTypeChecker = typeChecker;
        }
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      List<Argument> arguments = callExpression.arguments();

      String message = getMessageForConstructor(callExpression).orElse(null);
      if (message == null) {
        return;
      }

      ComprehensionExpression generator = getSingleGeneratorArg(arguments).orElse(null);
      if (generator == null) {
        return;
      }

      if (isGeneratorTransformingData(generator, callExpression.callee().typeV2())) {
        ctx.addIssue(callExpression.callee(), message);
      }
    });
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

  private boolean isGeneratorTransformingData(ComprehensionExpression generator, PythonType methodType) {
    if (hasMultipleNestedClauses(generator.comprehensionFor())) {
      return true;
    }
    if (dictTypeChecker.check(methodType) == TriBool.TRUE) {
      return isDictTransformingData(generator);
    } else {
      return isListOrSetComprehensionTransformingData(generator);
    }
  }

  private static boolean hasMultipleNestedClauses(ComprehensionClause clause) {
    return clause.nestedClause() != null;
  }

  private static boolean isDictTransformingData(ComprehensionExpression generator) {
    Expression resultExpr = generator.resultExpression();
    Expression loopVarExpr = generator.comprehensionFor().loopExpression();

    var resultTuple = getTwoNamedTuple(resultExpr);
    var varTuple = getTwoNamedTuple(loopVarExpr);

    if (resultTuple != null && varTuple != null) {
      return !resultTuple.equals(varTuple);
    }

    return true;
  }

  private static TwoNamedTuple getTwoNamedTuple(Expression expr) {
    if (expr instanceof Tuple tuple && tuple.elements().size() == 2) {
      var elements = tuple.elements();
      if (elements.get(0) instanceof Name name1 && elements.get(1) instanceof Name name2) {
        return new TwoNamedTuple(name1.symbolV2(), name2.symbolV2());
      }
    }
    return null;
  }

  private record TwoNamedTuple(@Nullable SymbolV2 symbol1, @Nullable SymbolV2 symbol2) {
  }

  private static boolean isListOrSetComprehensionTransformingData(ComprehensionExpression generator) {
    Expression elementExpr = generator.resultExpression();
    Expression loopVarExpr = generator.comprehensionFor().loopExpression();

    if (elementExpr instanceof Name elementName && loopVarExpr instanceof Name loopVarName) {
      return elementName.symbolV2() != loopVarName.symbolV2();
    }

    return true;
  }
}
