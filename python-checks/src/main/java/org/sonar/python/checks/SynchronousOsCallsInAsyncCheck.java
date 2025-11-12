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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7489")
public class SynchronousOsCallsInAsyncCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Use a thread executor to wrap blocking OS calls in this async function.";
  private static final String SECONDARY_MESSAGE = "This function is async.";
  private static final String TRIO_QUICK_FIX = "Wrap with \"await trio.thread.executor\".";
  private static final String ANYIO_QUICK_FIX = "Wrap with \"await anyio.thread.executor\".";
  
  private static final String TRIO = "trio";
  private static final String ANYIO = "anyio";
  private static final String THREAD_RUN_SYNC_FORMAT = "%s.to_thread.run_sync";

  private static final Set<String> OS_BLOCKING_CALLS = Set.of(
    "os.wait",
    "os.waitpid",
    "os.waitid"
  );

  private TypeCheckMap<Object> syncOsCallsTypeChecks;
  private final Map<String, String> asyncLibraryAliases = new HashMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_NAME, this::trackAsyncLibraryImports);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkSyncOsCallsInAsync);
  }

  private void setupTypeChecks(SubscriptionContext ctx) {
    syncOsCallsTypeChecks = new TypeCheckMap<>();
    var object = new Object();
    OS_BLOCKING_CALLS.forEach(path ->
      syncOsCallsTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName(path), object));
    asyncLibraryAliases.clear();
  }

  private void trackAsyncLibraryImports(SubscriptionContext ctx) {
    ImportName importName = (ImportName) ctx.syntaxNode();
    for (AliasedName module : importName.modules()) {
      List<Name> names = module.dottedName().names();
      if (names.size() > 1) {
        continue;
      }
      String moduleName = names.get(0).name();
      Name alias = module.alias();
      String moduleAlias = alias != null ? alias.name() : moduleName;
      if (TRIO.equals(moduleName) || ANYIO.equals(moduleName)) {
        asyncLibraryAliases.put(moduleName, moduleAlias);
      }
    }
  }

  private void checkSyncOsCallsInAsync(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    var asyncToken = TreeUtils.asyncTokenOfEnclosingFunction(callExpression).orElse(null);
    if (asyncToken == null) {
      return;
    }

    syncOsCallsTypeChecks.getOptionalForType(callExpression.callee().typeV2())
      .ifPresent(object -> {
        var issue = ctx.addIssue(callExpression.callee(), MESSAGE).secondary(asyncToken, SECONDARY_MESSAGE);
        addQuickFixes(issue, callExpression);
      });
  }

  private void addQuickFixes(PreciseIssue issue, CallExpression callExpression) {
    Optional<String> calleeDottedName = TreeUtils.stringValueFromNameOrQualifiedExpression(callExpression.callee());
    
    if (calleeDottedName.isPresent()) {
      if (asyncLibraryAliases.containsKey(TRIO)) {
        String alias = asyncLibraryAliases.get(TRIO);
        issue.addQuickFix(createThreadExecutorQuickFix(callExpression, String.format(THREAD_RUN_SYNC_FORMAT, alias),
          calleeDottedName.get(), TRIO_QUICK_FIX));
      }

      if (asyncLibraryAliases.containsKey(ANYIO)) {
        String alias = asyncLibraryAliases.get(ANYIO);
        issue.addQuickFix(createThreadExecutorQuickFix(callExpression, String.format(THREAD_RUN_SYNC_FORMAT, alias),
          calleeDottedName.get(), ANYIO_QUICK_FIX));
      }
    }
  }

  private static PythonQuickFix createThreadExecutorQuickFix(CallExpression callExpression,
    String executorPrefix,
    String calleeDottedName,
    String message) {
    return PythonQuickFix.newQuickFix(message)
      .addTextEdit(TextEditUtils.replace(callExpression.callee(), "await " + executorPrefix))
      .addTextEdit(TextEditUtils.insertAfter(callExpression.leftPar(),
        calleeDottedName + (callExpression.arguments().isEmpty() ? "" : ", ")))
      .build();
  }
}

