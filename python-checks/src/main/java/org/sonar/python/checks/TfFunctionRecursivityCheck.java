/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6908")
public class TfFunctionRecursivityCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Make sure to avoid recursive calls in this function.";
  private static final String SECONDARY_MESSAGE = "Recursive call is here.";
  private static final TypeMatcher TENSORFLOW_FUNCTION_MATCHER = TypeMatchers.withFQN("tensorflow.function");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, TfFunctionRecursivityCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext context) {
    FunctionDef functionDef = (FunctionDef) context.syntaxNode();
    if (!hasTensorflowFunctionDecorator(functionDef, context)) {
      return;
    }
    SymbolV2 functionSymbol = functionDef.name().symbolV2();
    if (functionSymbol == null || functionSymbol.getSingleBindingUsage().isEmpty()
      || !(functionDef.name().typeV2() instanceof FunctionType functionType)) {
      return;
    }
    CallCollector collector = new CallCollector(functionSymbol, functionType);
    functionDef.accept(collector);
    if (collector.expressionList.isEmpty()) {
      return;
    }
    var issue = context.addIssue(functionDef.name(), MESSAGE);
    collector.expressionList.forEach(call -> issue.secondary(call, SECONDARY_MESSAGE));
  }

  private static boolean hasTensorflowFunctionDecorator(FunctionDef functionDef, SubscriptionContext context) {
    return functionDef.decorators().stream()
      .anyMatch(decorator -> TENSORFLOW_FUNCTION_MATCHER.isTrueFor(decorator.expression(), context));
  }

  private static class CallCollector extends BaseTreeVisitor {
    private final SymbolV2 originalSymbol;
    private final FunctionType originalType;
    List<Expression> expressionList = new ArrayList<>();

    private CallCollector(SymbolV2 originalSymbol, FunctionType originalType) {
      this.originalSymbol = originalSymbol;
      this.originalType = originalType;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Expression callee = callExpression.callee();
      if ((callee instanceof Name name && originalSymbol.equals(name.symbolV2()))
        || (callee instanceof QualifiedExpression && originalType.equals(callee.typeV2()))) {
        expressionList.add(callExpression.callee());
      }
      super.visitCallExpression(callExpression);
    }
  }
}
