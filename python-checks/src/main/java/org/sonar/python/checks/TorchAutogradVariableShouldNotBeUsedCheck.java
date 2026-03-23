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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S6979")
public class TorchAutogradVariableShouldNotBeUsedCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Replace this call with a call to \"torch.tensor\".";
  // torch.autograd.Variable has no stubs, so the type resolves as UnresolvedImportType. Use withFQN to match on the FQN directly.
  private static final TypeMatcher TORCH_AUTOGRAD_VARIABLE = TypeMatchers.withFQN("torch.autograd.Variable");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      if (TORCH_AUTOGRAD_VARIABLE.isTrueFor(callExpression.callee(), ctx)) {
        ctx.addIssue(callExpression.callee(), MESSAGE);
      }
    });
  }
}
