/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nonnull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6908")
public class TfFunctionRecursivityCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this recursive call.";
  private static final String SECONDARY_MESSAGE = "Recursive call is here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, TfFunctionRecursivityCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext context) {
    FunctionDef functionDef = (FunctionDef) context.syntaxNode();
    if (!isTfFunction(functionDef)) {
      return;
    }
    var selfSymbol = TreeUtils.getFunctionSymbolFromDef(functionDef);
    if (selfSymbol == null) {
      return;
    }
    var collector = new CallCollector(selfSymbol);
    context.syntaxNode().accept(collector);
    if (collector.expressionList.isEmpty()) {
      return;
    }
    var issue = context.addIssue(functionDef.name(), MESSAGE);
    collector.expressionList.forEach(call -> issue.secondary(call, SECONDARY_MESSAGE));
  }

  private static boolean isTfFunction(FunctionDef functionDefinition) {
    return functionDefinition.decorators().stream()
      .map(Decorator::expression).map(TreeUtils::getSymbolFromTree)
      .filter(Optional::isPresent).map(Optional::get)
      .map(Symbol::fullyQualifiedName)
      .filter(Objects::nonNull)
      .anyMatch("tensorflow.function"::equals);
  }

  private static class CallCollector extends BaseTreeVisitor {
    private final Symbol originalSymbol;
    List<Expression> expressionList = new ArrayList<>();

    Set<FunctionDef> visited = new HashSet<>();

    private CallCollector(@Nonnull Symbol originalSymbol) {
      this.originalSymbol = originalSymbol;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Symbol symbol = TreeUtils.getSymbolFromTree(callExpression.callee()).orElse(null);
      if (symbol == null) {
        return;
      }
      if (symbol.equals(originalSymbol)) {
        expressionList.add(callExpression.callee());
      }
      if (symbol.is(Symbol.Kind.FUNCTION)) {
        symbol.usages().stream().filter(usage -> usage.kind() == (Usage.Kind.FUNC_DECLARATION)).forEach(usage -> usage.tree().parent().accept(this));
      }
      super.visitCallExpression(callExpression);
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      if (visited.contains(functionDef)) {
        return;
      }
      visited.add(functionDef);
      super.visitFunctionDef(functionDef);
    }
  }
}
