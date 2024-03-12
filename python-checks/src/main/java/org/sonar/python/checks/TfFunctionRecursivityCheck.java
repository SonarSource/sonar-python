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
import java.util.Optional;
import javax.annotation.Nonnull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6908")
public class TfFunctionRecursivityCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this recursive call.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TfFunctionRecursivityCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    var containingFunctions = new ArrayList<FunctionDef>();
    var currentFunction = TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.FUNCDEF);
    if (currentFunction == null) {
      return;
    }
    while (currentFunction != null) {
      containingFunctions.add((FunctionDef) currentFunction);
      currentFunction = TreeUtils.firstAncestorOfKind(currentFunction, Tree.Kind.FUNCDEF);
    }

    if (containingFunctions.stream().noneMatch(TfFunctionRecursivityCheck::isTfFunction)) {
      return;
    }

    Symbol calleeSymbol = callExpression.calleeSymbol();
    Optional.ofNullable(calleeSymbol)
      .filter(symbol -> symbol.is(Symbol.Kind.FUNCTION))
      .filter(symbol -> symbol == (TreeUtils.getFunctionSymbolFromDef(containingFunctions.get(0))))
      .ifPresent(symbol -> context.addIssue(callExpression, MESSAGE));
  }

  private static boolean isTfFunction(FunctionDef functionDefinition) {
    return Optional.of(functionDefinition).map(fd -> (FunctionDefImpl) fd).map(FunctionDefImpl::functionSymbol).map(TfFunctionRecursivityCheck::isTfFunction).orElse(false);
  }

  private static boolean isTfFunction(@Nonnull FunctionSymbol functionSymbol) {
    return functionSymbol.decorators().stream().anyMatch("tf.function"::equals);
  }
}
