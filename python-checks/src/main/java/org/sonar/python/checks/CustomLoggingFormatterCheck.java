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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.StringLiteralImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7946")
public class CustomLoggingFormatterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use built-in logging formatting instead of using custom string formatting.";

  TypeCheckMap<Integer> loggingMethodCallsCheck = new TypeCheckMap<>();
  TypeCheckBuilder strFormatCheck;
  TypeCheckBuilder strCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, this::initializeState);
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, this::checkCustomFormatter);
  }

  private void initializeState(SubscriptionContext ctx) {
    loggingMethodCallsCheck.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("logging.debug"), 0);
    loggingMethodCallsCheck.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("logging.info"), 0);
    loggingMethodCallsCheck.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("logging.error"), 0);
    loggingMethodCallsCheck.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("logging.warning"), 0);
    loggingMethodCallsCheck.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("logging.warn"), 0);
    loggingMethodCallsCheck.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("logging.critical"), 0);
    loggingMethodCallsCheck.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("logging.exception"), 0);
    loggingMethodCallsCheck.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("logging.log"), 1);
    strFormatCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("str.format");
    strCheck = ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("str");
  }

  private void checkCustomFormatter(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();
    Optional<Integer> argumentIndex = loggingMethodCallsCheck.getOptionalForType(call.callee().typeV2());
    if (argumentIndex.isEmpty()) {
      return;
    }
    var messageArgument = TreeUtils.nthArgumentOrKeyword(argumentIndex.get(), "msg", call.arguments());
    if (messageArgument == null) {
      return;
    }
    if (isArgumentAFormattedString(messageArgument.expression())) {
      ctx.addIssue(messageArgument, MESSAGE);
    }
  }

  private boolean isArgumentAFormattedString(Expression argument) {
    return (argument instanceof StringLiteralImpl stringLiteral && stringLiteral.isInterpolated() && hasAFormattedExpression(stringLiteral)) ||
      (argument instanceof CallExpression callExpression && strFormatCheck.check(callExpression.callee().typeV2()).isTrue()) ||
      (argument instanceof BinaryExpression expr && strCheck.check(expr.typeV2()).isTrue());
  }

  private static boolean hasAFormattedExpression(StringLiteralImpl stringLiteral) {
    return stringLiteral.stringElements().stream().anyMatch(element -> !element.formattedExpressions().isEmpty());
  }
}
