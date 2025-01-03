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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6925")
public class TfGatherDeprecatedValidateIndicesCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Don't set the `validate_indices` argument, it is deprecated.";
  private static final String FQN = "tensorflow.gather";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TfGatherDeprecatedValidateIndicesCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    Optional.ofNullable(((CallExpression) context.syntaxNode()).calleeSymbol())
      .filter(symbol -> FQN.equals(symbol.fullyQualifiedName()))
      .map(callExpression -> TreeUtils.nthArgumentOrKeyword(2, "validate_indices", ((CallExpression) context.syntaxNode()).arguments()))
      .ifPresent(argument -> context.addIssue(argument, MESSAGE));
  }
}
