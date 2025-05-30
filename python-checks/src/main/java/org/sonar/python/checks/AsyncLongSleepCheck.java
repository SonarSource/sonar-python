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
import java.util.Optional;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.checks.hotspots.CommonValidationUtils;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7486")
public class AsyncLongSleepCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this call with \"%s.sleep_forever()\" as the sleep duration exceeds 24 hours.";
  private static final String SECONDARY_MESSAGE = "This function is async.";
  private static final String QUICK_FIX_MESSAGE = "Replace with %s";
  private static final int SECONDS_IN_DAY = 86400;

  private TypeCheckBuilder isTrioSleepCall;
  private TypeCheckBuilder isAnyioSleepCall;
  private final Map<String, String> asyncLibraryAliases = new HashMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_NAME, this::checkImportName);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkAsyncLongSleep);
  }

  private void setupCheck(SubscriptionContext ctx) {
    isTrioSleepCall = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("trio.sleep");
    isAnyioSleepCall = ctx.typeChecker().typeCheckBuilder().isTypeWithName("anyio.sleep");
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
      Name aliasName = module.alias();
      String alias = aliasName != null ? aliasName.name() : moduleName;
      if ("trio".equals(moduleName) || "anyio".equals(moduleName)) {
        asyncLibraryAliases.put(moduleName, alias);
      }
    }
  }

  private void checkAsyncLongSleep(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Expression callee = callExpression.callee();

    String libraryName;
    if (isTrioSleepCall.check(callee.typeV2()) == TriBool.TRUE) {
      libraryName = "trio";
    } else if (isAnyioSleepCall.check(callee.typeV2()) == TriBool.TRUE) {
      libraryName = "anyio";
    } else {
      return;
    }

    Token asyncToken = TreeUtils.asyncTokenOfEnclosingFunction(callExpression).orElse(null);
    if (asyncToken == null) {
      return;
    }

    Expression durationExpr = extractDurationExpression(callExpression, "trio".equals(libraryName) ? "seconds" : "delay").orElse(null);
    if (durationExpr == null) {
      return;
    }
    if (CommonValidationUtils.isMoreThan(durationExpr, SECONDS_IN_DAY)) {
      String message = String.format(MESSAGE, libraryName);
      var issue = context.addIssue(callExpression, message)
        .secondary(asyncToken, SECONDARY_MESSAGE);
      createQuickFix(libraryName, callExpression)
        .ifPresent(issue::addQuickFix);
    }
  }

  private static Optional<Expression> extractDurationExpression(CallExpression callExpression, String paramName) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, paramName, callExpression.arguments())).map(RegularArgument::expression);
  }

  private Optional<PythonQuickFix> createQuickFix(String libraryName, CallExpression callExpression) {
    String alias = asyncLibraryAliases.get(libraryName);
    if (alias == null) {
      return Optional.empty();
    }
    String replacement = alias + ".sleep_forever()";
    String quickFixMsg = String.format(QUICK_FIX_MESSAGE, replacement);
    return Optional.of(
      PythonQuickFix.newQuickFix(quickFixMsg)
        .addTextEdit(TextEditUtils.replace(callExpression, replacement))
        .build());
  }
}
