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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7609")
public class AwsCustomMetricNamespaceCheck extends PythonSubscriptionCheck {

  public static final String NAMESPACE_ARGUMENT_KEYWORD = "Namespace";

  private TypeCheckBuilder isBoto3ClientPutMetricDataCheck;
  private TypeCheckBuilder isAioBotocoreClientPutMetricDataCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initializeCheck(SubscriptionContext ctx) {
    isBoto3ClientPutMetricDataCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("botocore.client.BaseClient.put_metric_data");
    isAioBotocoreClientPutMetricDataCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("aiobotocore.client.AioBaseClient.put_metric_data");
  }

  private void check(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    if (isSensitiveCall(callExpression)) {
      RegularArgument argument = TreeUtils.argumentByKeyword(NAMESPACE_ARGUMENT_KEYWORD, callExpression.arguments());
      if (argument != null && isInvalidArgumentValue(argument)) {
        ctx.addIssue(argument, "Do not use AWS reserved namespace that begins with 'AWS/' for custom metrics.");
      }
    }
  }

  private boolean isSensitiveCall(CallExpression callExpression) {
    PythonType type = TreeUtils.inferSingleAssignedExpressionType(callExpression.callee());
    return isBoto3ClientPutMetricDataCheck.check(type).isTrue() || isAioBotocoreClientPutMetricDataCheck.check(type).isTrue();
  }

  private static boolean isInvalidArgumentValue(RegularArgument regularArgument) {
    return Optional.of(regularArgument)
      .map(RegularArgument::expression)
      .flatMap(Expressions::ifNameGetSingleAssignedNonNameValue)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .map(StringLiteral::trimmedQuotesValue)
      .filter(value -> value.startsWith("AWS/"))
      .isPresent();
  }

}
