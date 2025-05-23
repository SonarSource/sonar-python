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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7489")
public class SynchronousOsCallsInAsyncCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Use a thread executor to wrap blocking OS calls in this async function.";
  private static final String SECONDARY_MESSAGE = "This function is async.";

  private static final Set<String> OS_BLOCKING_CALLS = Set.of(
    "os.wait",
    "os.waitpid",
    "os.waitid"
  );

  private final TypeCheckMap<Object> syncOsCallsTypeChecks = new TypeCheckMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkSyncOsCallsInAsync);
  }

  private void setupTypeChecks(SubscriptionContext ctx) {
    var object = new Object();
    OS_BLOCKING_CALLS.forEach(path -> 
      syncOsCallsTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName(path), object));
  }

  private void checkSyncOsCallsInAsync(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    var asyncToken = TreeUtils.asyncTokenOfEnclosingFunction(callExpression).orElse(null);
    if (asyncToken == null) {
      return;
    }

    syncOsCallsTypeChecks.getOptionalForType(callExpression.callee().typeV2())
      .ifPresent(object -> ctx.addIssue(callExpression.callee(), MESSAGE).secondary(asyncToken, SECONDARY_MESSAGE));
  }
}
