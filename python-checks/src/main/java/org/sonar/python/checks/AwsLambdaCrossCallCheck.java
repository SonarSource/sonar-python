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
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.AwsLambdaChecksUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S6246")
public class AwsLambdaCrossCallCheck extends PythonSubscriptionCheck {

  public static final String INVOCATION_TYPE_ARGUMENT_KEYWORD = "InvocationType";
  public static final String REQUEST_RESPONSE_INVOCATION_TYPE_ARGUMENT_VALUE = "RequestResponse";
  private TypeCheckBuilder isBoto3ClientCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initializeCheck(SubscriptionContext ctx) {
    isBoto3ClientCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("botocore.client.BaseClient.invoke");
  }

  private void check(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    if (isBoto3ClientCheck.check(TreeUtils.inferSingleAssignedExpressionType(callExpression.callee())).isTrue()
        && TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.FUNCDEF) instanceof FunctionDef functionDef
        && AwsLambdaChecksUtils.isLambdaHandler(ctx, functionDef)
        && hasInvalidArgumentValue(callExpression)) {
      ctx.addIssue(callExpression, "Avoid synchronous calls to other lambdas");
    }
  }

  private static boolean hasInvalidArgumentValue(CallExpression callExpression) {
    return Optional.ofNullable(TreeUtils.argumentByKeyword(INVOCATION_TYPE_ARGUMENT_KEYWORD, callExpression.arguments()))
      .map(RegularArgument::expression)
      .flatMap(Expressions::ifNameGetSingleAssignedNonNameValue)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .map(StringLiteral::trimmedQuotesValue)
      .filter(REQUEST_RESPONSE_INVOCATION_TYPE_ARGUMENT_VALUE::equals)
      .isPresent();
  }
}
