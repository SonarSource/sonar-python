/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import static org.sonar.python.checks.hotspots.CommonValidationUtils.isEqualTo;

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
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7491")
public class SleepZeroInAsyncCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use %s instead of %s.";
  private static final String SECONDARY_MESSAGE = "This function is async.";
  private static final String QUICK_FIX_MESSAGE = "Replace with %s";

  private TypeCheckMap<MessageHolder> asyncSleepFunctions;
  private final Map<String, String> asyncLibraryAliases = new HashMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, this::initializeAnalysis);
    context.registerSyntaxNodeConsumer(Kind.IMPORT_NAME, this::handleImportName);
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, this::checkCallExpr);
  }

  private void initializeAnalysis(SubscriptionContext context) {
    asyncSleepFunctions = TypeCheckMap.ofEntries(
      Map.entry(context.typeChecker().typeCheckBuilder().isTypeWithFqn("trio.sleep"),
        new MessageHolder("trio.sleep", "%s.lowlevel.checkpoint()", "seconds", "trio")),
      Map.entry(context.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("anyio.sleep"),
        new MessageHolder("anyio.sleep", "%s.lowlevel.checkpoint()", "delay", "anyio")));

    asyncLibraryAliases.clear();
  }

  private void handleImportName(SubscriptionContext context) {
    var importName = (ImportName) context.syntaxNode();
    importName.modules().forEach(this::trackModuleImport);
  }

  private void trackModuleImport(AliasedName aliasedName) {
    List<Name> names = aliasedName.dottedName().names();
    if (names.size() > 1) {
      return;
    }

    String moduleName = names.get(0).name();
    if ("trio".equals(moduleName) || "anyio".equals(moduleName)) {
      Name alias = aliasedName.alias();
      String moduleAlias = alias != null ? alias.name() : moduleName;
      asyncLibraryAliases.put(moduleName, moduleAlias);
    }
  }

  private void checkCallExpr(SubscriptionContext context) {
    var callExpr = (CallExpression) context.syntaxNode();
    var asyncKeyword = TreeUtils.asyncTokenOfEnclosingFunction(callExpr).orElse(null);
    if (asyncKeyword == null) {
      return;
    }

    var callee = callExpr.callee();
    asyncSleepFunctions.getOptionalForType(callee.typeV2()).ifPresent(
      messageHolder -> handleCallExpr(context, messageHolder, callExpr, asyncKeyword, asyncLibraryAliases));
  }

  private static void handleCallExpr(SubscriptionContext context, MessageHolder messageHolder, CallExpression callExpr, Tree asyncKeyword,
    Map<String, String> asyncLibraryAliases) {
    if (!isZero(callExpr, messageHolder.keywordArgumentName())) {
      return;
    }

    var moduleAlias = asyncLibraryAliases.getOrDefault(messageHolder.libraryName(), messageHolder.libraryName());
    var formattedReplacement = messageHolder.replacement().formatted(moduleAlias);

    var message = String.format(MESSAGE, formattedReplacement, messageHolder.fqn());
    var issue = context.addIssue(callExpr, message);
    issue.secondary(asyncKeyword, SECONDARY_MESSAGE);

    createQuickFix(messageHolder, callExpr, formattedReplacement, asyncLibraryAliases).ifPresent(issue::addQuickFix);
  }

  private static Optional<PythonQuickFix> createQuickFix(MessageHolder messageHolder, CallExpression callExpr, String formattedReplacement,
    Map<String, String> asyncLibraryAliases) {
    if (!asyncLibraryAliases.containsKey(messageHolder.libraryName())) {
      return Optional.empty();
    }

    var quickFixMessage = String.format(QUICK_FIX_MESSAGE, formattedReplacement);
    var quickFix = PythonQuickFix.newQuickFix(quickFixMessage)
      .addTextEdit(TextEditUtils.replace(callExpr, formattedReplacement))
      .build();
    return Optional.of(quickFix);
  }

  private static boolean isZero(CallExpression callExpr, String keywordArgumentName) {
    var argument = TreeUtils.nthArgumentOrKeyword(0, keywordArgumentName, callExpr.arguments());
    if (argument == null) {
      return false;
    }
    return isEqualTo(argument.expression(), 0);
  }

  record MessageHolder(String fqn, String replacement, String keywordArgumentName, String libraryName) {
  }
}
