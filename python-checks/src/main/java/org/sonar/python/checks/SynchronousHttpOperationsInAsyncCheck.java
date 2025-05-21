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

@Rule(key = "S7499")
public class SynchronousHttpOperationsInAsyncCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Use an async HTTP client in this async function instead of a synchronous one.";
  private static final String SECONDARY_MESSAGE = "This function is async.";

  private static final Set<String> IMPORT_PATHS = Set.of(
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.delete",
    "requests.head",
    "requests.options",
    "requests.patch",
    "requests.sessions.Session.get",
    "requests.sessions.Session.post",
    "requests.sessions.Session.put",
    "requests.sessions.Session.delete",
    "requests.sessions.Session.head",
    "requests.sessions.Session.options",
    "requests.sessions.Session.patch",
    "urllib3.PoolManager",
    "urllib3.PoolManager.request");

  private static final Set<String> IMPORT_PATHS_FQN = Set.of(
    "httpx.get",
    "httpx.post",
    "httpx.put",
    "httpx.delete",
    "httpx.head",
    "httpx.options",
    "httpx.patch");

  private final TypeCheckMap<Object> syncHttpTypeChecks = new TypeCheckMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkSyncHttpInAsync);
  }

  private void setupTypeChecks(SubscriptionContext ctx) {
    var object = new Object();
    IMPORT_PATHS.forEach(path -> syncHttpTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName(path), object));
    IMPORT_PATHS_FQN.forEach(path -> syncHttpTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(path), object));
  }

  private void checkSyncHttpInAsync(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    var asyncToken = TreeUtils.asyncTokenOfEnclosingFunction(callExpression).orElse(null);
    if (asyncToken == null) {
      return;
    }

    syncHttpTypeChecks.getOptionalForType(callExpression.callee().typeV2())
      .ifPresent(object -> ctx.addIssue(callExpression.callee(), MESSAGE).secondary(asyncToken, SECONDARY_MESSAGE));
  }
}
