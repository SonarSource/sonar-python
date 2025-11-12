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
package org.sonar.python.checks.cdk;

import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.checks.utils.AwsLambdaChecksUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7618")
public class NetworkCallsWithoutTimeoutsInLambdaCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Set an explicit timeout for this network call to prevent hanging executions in Lambda functions.";

  private static final int BOTO3_CLIENT_RESOURCE_CONFIG_NTH_ARGUMENT = 9;

  private static final Set<String> REQUESTS_METHODS = Set.of("get", "post", "put", "delete", "head", "options", "patch",
      "request");

  private static final Set<String> BOTO3_ENTRY_FUNCTIONS = Set.of(
      "boto3.client", "boto3.resource",
      "boto3.session.Session.client", "boto3.session.Session.resource");

  private TypeCheckMap<Object> requestsTypeChecks;
  private TypeCheckMap<Object> boto3EntryFunctionTypeChecks;
  private TypeCheckBuilder configConstructorTypeCheck;

  private boolean isLambdaHandlerInThisFile = false;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeTypeCheckMaps);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkNetworkCall);
  }

  private void initializeTypeCheckMaps(SubscriptionContext ctx) {
    isLambdaHandlerInThisFile = AwsLambdaChecksUtils.isLambdaHandlerInThisFile(ctx, ctx.syntaxNode());

    requestsTypeChecks = new TypeCheckMap<>();

    Object marker = new Object();

    for (String method : REQUESTS_METHODS) {
      requestsTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("requests." + method), marker);
      requestsTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName("requests.sessions.Session." + method),
          marker);
    }

    boto3EntryFunctionTypeChecks = new TypeCheckMap<>();
    for (String entryFunction : BOTO3_ENTRY_FUNCTIONS) {
      boto3EntryFunctionTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeWithName(entryFunction), marker);
    }

    configConstructorTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("botocore.config.Config");
  }

  private void checkNetworkCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    if (!isLambdaHandlerInThisFile) {
      return;
    }

    checkRequestsCall(ctx, callExpression);
    checkBoto3Call(ctx, callExpression);
  }

  private void checkRequestsCall(SubscriptionContext ctx, CallExpression callExpression) {
    var type = callExpression.callee().typeV2();
    if (requestsTypeChecks.getOptionalForType(type).isPresent()) {
      checkTimeoutParameter(ctx, callExpression);
    }
  }

  private static void checkTimeoutParameter(SubscriptionContext ctx, CallExpression callExpression) {
    var timeoutArg = CdkUtils.getArgument(ctx, callExpression, "timeout");
    if (timeoutArg.isEmpty()) {
      ctx.addIssue(callExpression.callee(), MESSAGE);
    } else {
      timeoutArg.get().addIssueIf(CdkPredicate.isNone(), MESSAGE, callExpression);
    }
  }

  private void checkBoto3Call(SubscriptionContext ctx, CallExpression callExpression) {
    var type = callExpression.callee().typeV2();

    if (!isBoto3ClientOrResource(type)) {
      return;
    }

    var configArg = getBotocoreConfigArgument(callExpression);
    if (configArg == null) {
      ctx.addIssue(callExpression.callee(), MESSAGE);
    } else {
      var configConstructor = getConfigArgumentConstructorCallExpr(configArg);
      // if configConstructor == null, we couldn't figure out what the config argument
      // is -> no issue raised to prevent FPs
      if (configConstructor != null && isInvalidConfigConstructorCall(configConstructor)) {
        ctx.addIssue(callExpression.callee(), MESSAGE);
      }
    }
  }

  private boolean isBoto3ClientOrResource(PythonType type) {
    return boto3EntryFunctionTypeChecks.getOptionalForType(type).isPresent();
  }

  @CheckForNull
  private static RegularArgument getBotocoreConfigArgument(CallExpression callExpression) {
    return TreeUtils.nthArgumentOrKeyword(BOTO3_CLIENT_RESOURCE_CONFIG_NTH_ARGUMENT, "config",
        callExpression.arguments());
  }

  @CheckForNull
  private static CallExpression getConfigArgumentConstructorCallExpr(RegularArgument arg) {
    Expression argExpr = arg.expression();
    if (argExpr instanceof Name argName) {
      return getSingleAssignedConfigConstructorCallExpr(argName).orElse(null);
    } else if (argExpr instanceof CallExpression callExpr) {
      return callExpr;
    }
    return null;
  }

  private static Optional<CallExpression> getSingleAssignedConfigConstructorCallExpr(Name configVarName) {
    return Expressions.singleAssignedNonNameValue(configVarName)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class));
  }

  private boolean isInvalidConfigConstructorCall(CallExpression callExpression) {
    return configConstructorTypeCheck.check(callExpression.callee().typeV2()).isTrue()
        && !hasArgumentWithName(callExpression, "read_timeout")
        && !hasArgumentWithName(callExpression, "connect_timeout");
  }

  private static boolean hasArgumentWithName(CallExpression callExpression, String name) {
    return TreeUtils.argumentByKeyword(name, callExpression.arguments()) != null;
  }
}
