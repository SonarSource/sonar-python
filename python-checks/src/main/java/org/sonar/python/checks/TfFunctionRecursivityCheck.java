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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6908")
public class TfFunctionRecursivityCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Make sure to avoid recursive calls in this function.";
  private static final String SECONDARY_MESSAGE = "Recursive call is here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, TfFunctionRecursivityCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext context) {
    FunctionDef functionDef = (FunctionDef) context.syntaxNode();
    if (!TreeUtils.isFunctionWithGivenDecoratorFQN(functionDef, "tensorflow.function")) {
      return;
    }
    FunctionSymbol functionSymbol = TreeUtils.getFunctionSymbolFromDef(functionDef);
    if (functionSymbol == null) {
      return;
    }
    CallCollector collector = new CallCollector(functionSymbol);
    functionDef.accept(collector);
    if (collector.expressionList.isEmpty()) {
      return;
    }
    var issue = context.addIssue(functionDef.name(), MESSAGE);
    collector.expressionList.forEach(call -> issue.secondary(call, SECONDARY_MESSAGE));
  }

  private static class CallCollector extends BaseTreeVisitor {
    private final Symbol originalSymbol;
    List<Expression> expressionList = new ArrayList<>();

    private CallCollector(Symbol originalSymbol) {
      this.originalSymbol = originalSymbol;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      TreeUtils.getSymbolFromTree(callExpression.callee())
        .filter(symbol -> symbol.equals(originalSymbol))
        .ifPresent(symbol -> expressionList.add(callExpression.callee()));
      super.visitCallExpression(callExpression);
    }
  }
}
