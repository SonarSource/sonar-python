/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.plugins.python.api.tree.Tree.Kind.CALL_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.COMPARISON;

@Rule(key = "S6660")
public class DirectTypeComparisonCheck extends PythonSubscriptionCheck {

  private static final Set<String> OPERATORS = Set.of("==", "!=");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(COMPARISON, ctx -> {
      BinaryExpression binaryExpression = (BinaryExpression) ctx.syntaxNode();
      if (!OPERATORS.contains(binaryExpression.operator().value())) return;
      checkBinaryExpression(ctx, binaryExpression);
    });
  }

  private static void checkBinaryExpression(SubscriptionContext ctx, BinaryExpression binaryExpression) {
    if (isDirectTypeComparison(binaryExpression.leftOperand(), binaryExpression.rightOperand())) {
      Token operator = binaryExpression.operator();
      ctx.addIssue(operator, "==".equals(operator.value()) ? "Use the `isinstance()` function here." : "Use `not isinstance()` here.");
    }
  }

  private static boolean isDirectTypeComparison(Expression lhs, Expression rhs) {
    return (isTypeBuiltinCall(lhs) && TreeUtils.getSymbolFromTree(rhs).filter(s -> s.is(CLASS)).isPresent())
      || (isTypeBuiltinCall(rhs) && TreeUtils.getSymbolFromTree(lhs).filter(s -> s.is(CLASS)).isPresent());
  }

  private static boolean isTypeBuiltinCall(Expression expression) {
    if (!expression.is(CALL_EXPR)) return false;
    Symbol calleeSymbol = ((CallExpression) expression).calleeSymbol();
    return calleeSymbol != null && "type".equals(calleeSymbol.fullyQualifiedName());
  }
}
