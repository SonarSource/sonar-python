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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6979")
public class TorchAutogradVariableShouldNotBeUsedCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Replace this call with a call to \"torch.tensor\".";
  private static final String TORCH_AUTOGRAD_VARIABLE = "torch.autograd.Variable";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && TORCH_AUTOGRAD_VARIABLE.equals(calleeSymbol.fullyQualifiedName())) {
        ctx.addIssue(callExpression.callee(), MESSAGE);
      }
    });
  }
}
