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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7501")
public class InputInAsyncCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TO_THREAD = "Wrap this call to input() with await %s.to_thread(input).";
  private static final String MESSAGE_TO_THREAD_RUN_SYNC = "Wrap this call to input() with await %s.to_thread.run_sync(input).";
  private static final String MESSAGE_FALLBACK = "Wrap this call to input() with the appropriate function from the asynchronous library.";
  private static final String SECONDARY_MESSAGE = "This function is async.";

  private static final String LIB_ASYNCIO = "asyncio";
  private static final String LIB_TRIO = "trio";
  private static final String LIB_ANYIO = "anyio";

  private TypeCheckBuilder isInputCall;
  private final Map<String, String> asyncLibraryAliases = new HashMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      isInputCall = ctx.typeChecker().typeCheckBuilder().isTypeWithName("input");
      asyncLibraryAliases.clear();
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_NAME, this::checkImportName);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkInputInAsync);
  }

  private void checkImportName(SubscriptionContext ctx) {
    ImportName importName = (ImportName) ctx.syntaxNode();
    for (AliasedName module : importName.modules()) {
      List<Name> names = module.dottedName().names();
      if (names.size() > 1) {
        continue;
      }
      String moduleName = names.get(0).name();
      Name alias = module.alias();
      String moduleAlias = alias != null ? alias.name() : moduleName;
      if (LIB_ASYNCIO.equals(moduleName) || LIB_TRIO.equals(moduleName) || LIB_ANYIO.equals(moduleName)) {
        asyncLibraryAliases.put(moduleName, moduleAlias);
      }
    }
  }

  private void checkInputInAsync(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Expression callee = callExpression.callee();
    if (isInputCall.check(callee.typeV2()) != TriBool.TRUE) {
      return;
    }
    TreeUtils.asyncTokenOfEnclosingFunction(callExpression)
      .ifPresent(asyncKeyword -> {
        String message = getMessage();
        context.addIssue(callee, message).secondary(asyncKeyword, SECONDARY_MESSAGE);
      });
  }

  private String getMessage() {
    if (asyncLibraryAliases.size() != 1) {
      return MESSAGE_FALLBACK;
    }
    var library = asyncLibraryAliases.keySet().iterator().next();
    if (LIB_ASYNCIO.equals(library)) {
      return String.format(MESSAGE_TO_THREAD, asyncLibraryAliases.get(LIB_ASYNCIO));
    } else if (LIB_TRIO.equals(library)) {
      return String.format(MESSAGE_TO_THREAD_RUN_SYNC, asyncLibraryAliases.get(LIB_TRIO));
    } else {
      return String.format(MESSAGE_TO_THREAD_RUN_SYNC, asyncLibraryAliases.get(LIB_ANYIO));
    }
  }
}
