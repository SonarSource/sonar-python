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
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7487")
public class SynchronousSubprocessOperationsInAsyncCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use an async subprocess call in this async function instead of a synchronous one.";
  private static final String SECONDARY_MESSAGE = "This function is async.";

  private static final Set<String> SYNC_SUBPROCESS_CALLS = Set.of(
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.getstatusoutput",
    "subprocess.getoutput",
    "os.system",
    "os.popen",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe");

  private TypeCheckMap<Object> allSyncTypeChecks;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeTypeCheckMap);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  private void initializeTypeCheckMap(SubscriptionContext ctx) {
    var marker = new Object();
    allSyncTypeChecks = new TypeCheckMap<>();
    SYNC_SUBPROCESS_CALLS.forEach(fqn -> allSyncTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName(fqn), marker));
  }

  private void checkCallExpression(SubscriptionContext ctx) {
    var call = (CallExpression) ctx.syntaxNode();
    var asyncToken = TreeUtils.asyncTokenOfEnclosingFunction(call).orElse(null);
    if (asyncToken == null) {
      return;
    }

    if (allSyncTypeChecks.getOptionalForType(call.callee().typeV2()).isPresent()) {
      ctx.addIssue(call.callee(), MESSAGE).secondary(asyncToken, SECONDARY_MESSAGE);
      return;
    }

    if (call.callee() instanceof QualifiedExpression member && allSyncTypeChecks.getOptionalForType(member.qualifier().typeV2()).isPresent()) {
      ctx.addIssue(call.callee(), MESSAGE).secondary(asyncToken, SECONDARY_MESSAGE);
    }
  }
}
