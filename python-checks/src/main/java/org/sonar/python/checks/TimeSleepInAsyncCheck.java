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
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7488")
public class TimeSleepInAsyncCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this call to time.sleep() with an asynchronous sleep function.";
  private static final String SECONDARY_MESSAGE = "This function is async.";
  private static final String SLEEP_QUICK_FIX = "Replace with %s.sleep()";

  private TypeCheckBuilder isTimeSleepCall;
  private final Map<String, String> asyncLibraryAliases = new HashMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_NAME, this::checkImportName);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkTimeSleepInAsync);
  }

  private void setupCheck(SubscriptionContext ctx) {
    isTimeSleepCall = ctx.typeChecker().typeCheckBuilder().isTypeWithName("time.sleep");
    asyncLibraryAliases.clear();
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
      if ("asyncio".equals(moduleName) || "trio".equals(moduleName) || "anyio".equals(moduleName)) {
        asyncLibraryAliases.put(moduleName, moduleAlias);
      }
    }
  }

  private void checkTimeSleepInAsync(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Expression callee = callExpression.callee();
    if (isTimeSleepCall.check(callee.typeV2()) != TriBool.TRUE) {
      return;
    }
    TreeUtils.asyncTokenOfEnclosingFunction(callExpression)
      .ifPresent(asyncKeyword -> {
        var issue = context.addIssue(callee, MESSAGE).secondary(asyncKeyword, SECONDARY_MESSAGE);
        addQuickFixes(issue, callExpression);
      });
  }

  private void addQuickFixes(PreciseIssue issue, CallExpression callExpression) {
    asyncLibraryAliases.forEach((library, alias) -> {
      String message = String.format(SLEEP_QUICK_FIX, alias);
      issue.addQuickFix(createQuickFix(callExpression, alias + ".sleep", message));
    });
  }

  private static PythonQuickFix createQuickFix(CallExpression callExpression, String replacementFunction, String message) {
    return PythonQuickFix.newQuickFix(message)
      .addTextEdit(TextEditUtils.replace(callExpression.callee(), replacementFunction))
      .build();
  }
}
