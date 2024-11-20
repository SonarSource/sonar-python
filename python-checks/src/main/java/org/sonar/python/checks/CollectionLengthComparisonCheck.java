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

import com.sonar.sslr.api.TokenType;
import java.util.EnumSet;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.plugins.python.api.symbols.Symbol;

import static org.sonar.python.checks.utils.Expressions.removeParentheses;

@Rule(key = "S3981")
public class CollectionLengthComparisonCheck extends PythonSubscriptionCheck {

  private static final EnumSet<PythonPunctuator> INVALID_OPERATORS =
    EnumSet.of(PythonPunctuator.LT, PythonPunctuator.GT_EQU);

  private static final EnumSet<PythonPunctuator> INVALID_REVERSE_OPERATORS =
    EnumSet.of(PythonPunctuator.GT, PythonPunctuator.LT_EQU);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.COMPARISON, ctx -> {
      BinaryExpression comparison = (BinaryExpression) ctx.syntaxNode();
      Expression left = removeParentheses(comparison.leftOperand());
      Expression right = removeParentheses(comparison.rightOperand());
      TokenType operator = comparison.operator().type();
      if ((isCallToLen(left) && isZero(right) && INVALID_OPERATORS.contains(operator))
        || (isCallToLen(right) && isZero(left) && INVALID_REVERSE_OPERATORS.contains(operator))) {
        ctx.addIssue(comparison, "The length of a collection is always \">=0\", so update this test to either \"==0\" or \">0\".");
      }
    });
  }

  private static boolean isZero(Expression expression) {
    return expression.is(Kind.NUMERIC_LITERAL) && "0".equals(((NumericLiteral) expression).valueAsString());
  }

  private static boolean isCallToLen(Expression expression) {
    if (expression.is(Kind.CALL_EXPR)) {
      Symbol calleeSymbol = ((CallExpression) expression).calleeSymbol();
      return calleeSymbol != null && "len".equals(calleeSymbol.fullyQualifiedName());
    }
    return false;
  }


}
