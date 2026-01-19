/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.checks.utils;

import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionCheck;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.ComprehensionClause;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.types.v2.TypeCheckBuilder;

public class IsComprehensionTransformedChecker {

  private TypeCheckBuilder dictTypeChecker = null;

  public IsComprehensionTransformedChecker(SubscriptionCheck.Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> dictTypeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("dict"));
  }

  public boolean isGeneratorTransformingData(ComprehensionExpression generator, PythonType methodType) {
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
