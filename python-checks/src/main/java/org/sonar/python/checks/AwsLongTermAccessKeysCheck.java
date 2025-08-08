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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7625")
public class AwsLongTermAccessKeysCheck extends PythonSubscriptionCheck {

  private static final String ACCESS_KEY_MESSAGE = "Make sure using long-term access keys is safe here.";
  private static final String SECRET_KEY_MESSAGE = "Make sure using long-term secret keys is safe here.";

  record ClientMethod(String fqn, int awsAccessKeyIdArgIndex) {
    public int awsSecretAccessKeyArgIndex() {
      return awsAccessKeyIdArgIndex + 1;
    }
  }

  private static final Set<ClientMethod> AWS_CLIENT_METHODS = Set.of(
    new ClientMethod("boto3.client", 6),
    new ClientMethod("boto3.resource", 6),
    new ClientMethod("boto3.session.Session.client", 6),
    new ClientMethod("boto3.session.Session.resource", 6),
    new ClientMethod("boto3.session.Session", 0)
  );

  private TypeCheckMap<ClientMethod> awsClientTypeChecks;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeTypeCheckMap);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  private void initializeTypeCheckMap(SubscriptionContext ctx) {
    awsClientTypeChecks = new TypeCheckMap<>();
    AWS_CLIENT_METHODS.forEach(clientMethod -> 
      awsClientTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName(clientMethod.fqn()), clientMethod));

  }

  private void checkCallExpression(SubscriptionContext ctx) {
    var call = (CallExpression) ctx.syntaxNode();
    ClientMethod clientMethod = awsClientTypeChecks.getForType(call.callee().typeV2());
    if (clientMethod != null) {
      RegularArgument accessKeyArgument = getAccessKeyArgument(clientMethod, call);
      RegularArgument secretKeyArgument = getSecretKeyArgument(clientMethod, call);
      if(accessKeyArgument != null && isLongTermKeyPassedAsArgument(accessKeyArgument)) {
        ctx.addIssue(accessKeyArgument, ACCESS_KEY_MESSAGE);
      } else if (secretKeyArgument != null && isLongTermKeyPassedAsArgument(secretKeyArgument)) {
        ctx.addIssue(secretKeyArgument, SECRET_KEY_MESSAGE);
      }
    }
  }

  private static RegularArgument getSecretKeyArgument(ClientMethod clientMethod, CallExpression call) {
    return TreeUtils.nthArgumentOrKeyword(clientMethod.awsSecretAccessKeyArgIndex(), "aws_secret_access_key", call.arguments());
  }

  private static RegularArgument getAccessKeyArgument(ClientMethod clientMethod, CallExpression call) {
    return TreeUtils.nthArgumentOrKeyword(clientMethod.awsAccessKeyIdArgIndex(), "aws_access_key_id", call.arguments());
  }

  private static boolean isLongTermKeyPassedAsArgument(RegularArgument argument) {
    // For the purpose of this rule, any string literal is considered a long-term key.
    // Short-term keys seem to usually use a different mechanism. Furthermore, there is no way to distinguish between the two statically
    Expression expression = argument.expression();
    return isStringLiteral(expression) || isAssignedValueString(expression);
  }

  private static boolean isAssignedValueString(Expression expression) {
    if(expression instanceof Name name) {
      Expression assignedValue = Expressions.singleAssignedValue(name);
      return assignedValue != null && isStringLiteral(assignedValue);
    }
    return false;
  }

  private static boolean isStringLiteral(Expression expr) {
    return expr.is(Tree.Kind.STRING_LITERAL);
  }
}
