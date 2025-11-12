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
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S6262")
public class AwsHardcodedRegionCheck extends PythonSubscriptionCheck {

  public static final String REGION_ARGUMENT_KEYWORD = "region_name";
  private static final String HARDCODED_REGION_ISSUE_MESSAGE = "AWS region should not be set with a hardcoded String";

  record ClientMethod(String fqn, int regionNameArgIndex) {
  }

  private static final Set<ClientMethod> AWS_CLIENT_METHODS = Set.of(
    new ClientMethod("boto3.client", 1),
    new ClientMethod("boto3.resource", 1),
    new ClientMethod("boto3.session.Session.client", 1),
    new ClientMethod("boto3.session.Session.resource", 1),
    new ClientMethod("boto3.session.Session", 3)
  );

  private TypeCheckMap<ClientMethod> isClientWithRegionNameParameterCheckMap;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initializeCheck(SubscriptionContext ctx) {
    isClientWithRegionNameParameterCheckMap = new TypeCheckMap<>();
    AWS_CLIENT_METHODS.forEach(
      method -> isClientWithRegionNameParameterCheckMap.put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(method.fqn()), method)
    );
  }

  private void check(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    checkSensitiveCall(callExpression)
      .flatMap(sensitiveClientCall -> getHarcodedRegionArgument(sensitiveClientCall, callExpression))
      .ifPresent(regularArgument -> ctx.addIssue(regularArgument, HARDCODED_REGION_ISSUE_MESSAGE));
  }

  private Optional<ClientMethod> checkSensitiveCall(CallExpression callExpression) {
    PythonType calleeType = TreeUtils.inferSingleAssignedExpressionType(callExpression.callee());
    return Optional.ofNullable(isClientWithRegionNameParameterCheckMap.getForType(calleeType));
  }

  private static Optional<RegularArgument> getHarcodedRegionArgument(ClientMethod clientMethod, CallExpression callExpression) {
    return TreeUtils.nthArgumentOrKeywordOptional(clientMethod.regionNameArgIndex(), REGION_ARGUMENT_KEYWORD, callExpression.arguments())
      .filter(AwsHardcodedRegionCheck::isHardcodedArgument);
  }

  private static boolean isHardcodedArgument(RegularArgument regularArgument) {
    Optional<Expression> expression = Expressions.ifNameGetSingleAssignedNonNameValue(regularArgument.expression());
    return expression.isPresent() && expression.get().is(Tree.Kind.STRING_LITERAL);
  }
}
