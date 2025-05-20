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

import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

import static org.sonar.python.checks.hotspots.CommonValidationUtils.isEqualTo;

@Rule(key = "S7491")
public class SleepZeroInAsyncCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use %s instead of %s.";
  private static final String SECONDARY_MESSAGE = "This function is async.";
  
  private TypeCheckMap<MessageHolder> asyncSleepFunctions;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, this::initializeTypeCheckMap);
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, this::checkCallExpr);
  }

  private void initializeTypeCheckMap(SubscriptionContext context) {
    asyncSleepFunctions = TypeCheckMap.ofEntries(
      Map.entry(context.typeChecker().typeCheckBuilder().isTypeWithFqn("trio.sleep"),
        new MessageHolder("trio.sleep", "trio.lowlevel.checkpoint()", "seconds")),
      Map.entry(context.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("anyio.sleep"),
        new MessageHolder("anyio.sleep", "anyio.lowlevel.checkpoint()", "delay")));
  }

  private void checkCallExpr(SubscriptionContext context) {
    var callExpr = (CallExpression) context.syntaxNode();
    var asyncKeyword = TreeUtils.asyncTokenOfEnclosingFunction(callExpr).orElse(null);
    if (asyncKeyword == null) {
      return;
    }

    var callee = callExpr.callee();
    asyncSleepFunctions.getOptionalForType(callee.typeV2()).ifPresent(
      messageHolder -> messageHolder.handleCallExpr(context, callExpr, asyncKeyword));
  }

  record MessageHolder(String fqn, String replacement, String keywordArgumentName) {
    public void handleCallExpr(SubscriptionContext ctx, CallExpression callExpr, Tree asyncKeyword) {
      if (!isZero(callExpr, keywordArgumentName)) {
        return;
      }
      var message = String.format(MESSAGE, replacement, fqn);
      var issue = ctx.addIssue(callExpr, message);
      issue.secondary(asyncKeyword, SECONDARY_MESSAGE);
    }

    private static boolean isZero(CallExpression callExpr, String keywordArgumentName) {
      var argument = TreeUtils.nthArgumentOrKeyword(0, keywordArgumentName, callExpr.arguments());
      if (argument == null) {
        return false;
      }
      return isEqualTo(argument.expression(), 0);
    }

  }
}
