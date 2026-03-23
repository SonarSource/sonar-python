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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FullyQualifiedNameHelper;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S5445")
public class TempFileCreationCheck extends PythonSubscriptionCheck {

  // os.tempnam and os.tmpnam were removed in Python 3 and have no stubs.
  // We use withFQN to match both qualified (os.tempnam) and direct import (from os import tempnam) forms,
  // as the callee gets typed as UnresolvedImportType("os.tempnam") in both cases.
  private static final TypeMatcher INSECURE_CALLS = TypeMatchers.any(
    TypeMatchers.isType("tempfile.mktemp"),
    TypeMatchers.withFQN("os.tempnam"),
    TypeMatchers.withFQN("os.tmpnam")
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TempFileCreationCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpr = (CallExpression) ctx.syntaxNode();
    Optional.of(callExpr.callee())
      .filter(callee -> INSECURE_CALLS.isTrueFor(callee, ctx))
      .flatMap(callee -> FullyQualifiedNameHelper.getFullyQualifiedName(callee.typeV2()))
      .ifPresent(name -> ctx.addIssue(callExpr.callee(), String.format("'%s' is insecure. Use 'tempfile.TemporaryFile' instead", name)));
  }
}
