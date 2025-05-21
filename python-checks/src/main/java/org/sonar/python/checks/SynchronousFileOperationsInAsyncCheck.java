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

import java.util.List;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7493")
public class SynchronousFileOperationsInAsyncCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use an asynchronous file API instead of synchronous %s() in this async function.";
  private static final String SECONDARY_MESSAGE = "This function is async.";

  private final TypeCheckMap<String> syncFileFunctions = new TypeCheckMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeTypeCheckMap);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  private static final List<String> SYNC_FILE_FUNCTIONS = List.of(
    "open",
    "os.open",
    "pathlib.Path.open",
    "codecs.open",
    "os.fdopen",
    "os.popen",
    "tempfile.TemporaryFile",
    "tempfile.NamedTemporaryFile",
    "tempfile.SpooledTemporaryFile",
    "gzip.open",
    "bz2.open",
    "lzma.open");

  private void initializeTypeCheckMap(SubscriptionContext context) {
    SYNC_FILE_FUNCTIONS.forEach(fqn -> syncFileFunctions.put(context.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName(fqn), fqn));
  }

  private void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    var asyncToken = TreeUtils.asyncTokenOfEnclosingFunction(callExpression).orElse(null);
    if (asyncToken == null) {
      return;
    }
    syncFileFunctions.getOptionalForType(callExpression.callee().typeV2()).ifPresent(
      functionName ->
        ctx.addIssue(callExpression, String.format(MESSAGE, functionName))
        .secondary(asyncToken, SECONDARY_MESSAGE));
  }
}
