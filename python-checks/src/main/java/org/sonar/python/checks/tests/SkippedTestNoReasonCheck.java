/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.tests;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S1607")
public class SkippedTestNoReasonCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Provide a reason for skipping this test.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, ctx -> {
      Decorator decorator = (Decorator) ctx.syntaxNode();

      checkQualifiedExpression(ctx, decorator, "unittest.case.skip");
      checkDecoratorCallExpressionWithEmptyStringArg(ctx, decorator, "unittest.case.skip");
      checkQualifiedExpression(ctx, decorator, "pytest.mark.skip");
      checkDecoratorCallExpressionWithEmptyStringArg(ctx, decorator, "pytest.mark.skip");
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      checkCallExpressionWithNoArgsOrEmptyStringArg(ctx, callExpression, "pytest.skip");
    });
  }

  private static void checkQualifiedExpression(SubscriptionContext ctx, Decorator decorator, String name) {
    if (decorator.expression().is(Tree.Kind.QUALIFIED_EXPR)) {
      QualifiedExpression qualifiedExpression = (QualifiedExpression) decorator.expression();

      Symbol symbol = qualifiedExpression.symbol();

      if (symbol == null) {
        return;
      }

      if (!name.equals(symbol.fullyQualifiedName())) {
        return;
      }

      if (decorator.arguments() == null) {
        ctx.addIssue(decorator, MESSAGE);
      }
    }
  }

  private static void checkDecoratorCallExpressionWithEmptyStringArg(SubscriptionContext ctx, Decorator decorator, String name) {
    if (!decorator.expression().is(Tree.Kind.CALL_EXPR)) return;

    CallExpression callExpression = (CallExpression) decorator.expression();
    checkCallExpressionWithNoArgsOrEmptyStringArg(ctx, callExpression, name);
  }

  private static void checkCallExpressionWithNoArgsOrEmptyStringArg(SubscriptionContext ctx, CallExpression callExpression, String name) {
    Symbol symbol = (callExpression.calleeSymbol());
    if (symbol == null) {
      return;
    }

    if (!name.equals(symbol.fullyQualifiedName())) {
      return;
    }

    if (callExpression.arguments().isEmpty()) {
      ctx.addIssue(callExpression, MESSAGE);
      return;
    }

    Argument arg = callExpression.arguments().get(0);
    if(!arg.is(Tree.Kind.REGULAR_ARGUMENT)) {
      return;
    }

    RegularArgument regularArg = (RegularArgument) arg;
    if(!regularArg.expression().is(Tree.Kind.STRING_LITERAL)) {
      return;
    }

    StringLiteral stringLiteral = (StringLiteral) regularArg.expression();

    if(stringLiteral.trimmedQuotesValue().equals("")) {
      ctx.addIssue(stringLiteral, MESSAGE);
    }
  }
}
