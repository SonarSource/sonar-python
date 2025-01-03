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
package org.sonar.python.checks.cdk;

import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.checks.cdk.CdkPredicate.hasArgument;
import static org.sonar.python.checks.cdk.CdkPredicate.hasIntervalArguments;
import static org.sonar.python.checks.cdk.CdkPredicate.isCallExpression;
import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isNumeric;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

public class UnrestrictedAdministrationCheckPartConnections extends AbstractCdkResourceCheck {
  private static final String MESSAGE_BAD_PEER = "Change this IP range to a subset of trusted IP addresses.";
  private static final String MESSAGE_BAD_METHOD = "Change this method for `allow_from` and set `other` to a subset of trusted IP addresses.";

  private static final String OTHER = "other";
  private static final String PORT_RANGE = "port_range";
  private static final Set<Long> ADMIN_PORTS = Set.of(22L, 3389L);

  // Predicates to detect sensitive arguments
  private static final Predicate<Expression> IS_SENSITIVE_PROTOCOL =
      isFqn("aws_cdk.aws_ec2.Protocol.ALL")
      .or(isFqn("aws_cdk.aws_ec2.Protocol.TCP"));
  private static final Predicate<Expression> IS_SENSITIVE_PORT =
    isCallExpression().and(
      isFqn("aws_cdk.aws_ec2.Port.all_tcp")
      .or(isFqn("aws_cdk.aws_ec2.Port.all_traffic"))
      .or(isFqn("aws_cdk.aws_ec2.Port.tcp").and(hasArgument("port", 0, isNumeric(ADMIN_PORTS))))
      .or(isFqn("aws_cdk.aws_ec2.Port.tcp_range").and(hasIntervalArguments("start_port", 0, "end_port", 1, ADMIN_PORTS)))
      .or(isFqn("aws_cdk.aws_ec2.Port").and(hasArgument("protocol", IS_SENSITIVE_PROTOCOL)).and(hasIntervalArguments("from_port", "to_port", ADMIN_PORTS)))
    );
  private static final Predicate<Expression> IS_SENSITIVE_PEER =
      isFqn("aws_cdk.aws_ec2.Peer.any_ipv4")
      .or(isFqn("aws_cdk.aws_ec2.Peer.any_ipv6"))
      .or(isFqn("aws_cdk.aws_ec2.Peer.ipv4").and(hasArgument("cidr_ip", 0, isString("0.0.0.0/0"))))
      .or(isFqn("aws_cdk.aws_ec2.Peer.ipv6").and(hasArgument("cidr_ip", 0, isString("::/0"))));

  @Override
  protected void registerFqnConsumer() {
    // aws_cdk.aws_ec2.Connections "allow from" methods call
    checkFqn("aws_cdk.aws_ec2.Connections.allow_from", checkPeerAndPortSensitivity(OTHER, PORT_RANGE));
    checkFqn("aws_cdk.aws_ec2.Connections.allow_from_any_ipv4", checkPortSensitivity(PORT_RANGE));
    checkFqn("aws_cdk.aws_ec2.Connections.allow_default_port_from", UnrestrictedAdministrationCheckPartConnections::checkPeerAndDefaultPortInConstructorCall);
    checkFqn("aws_cdk.aws_ec2.Connections.allow_default_port_from_any_ipv4", UnrestrictedAdministrationCheckPartConnections::checkDefaultPortInConstructorCall);

    // SecurityGroup.add_ingress_rule
    checkFqn("aws_cdk.aws_ec2.SecurityGroup.add_ingress_rule", checkPeerAndPortSensitivity("peer", "connection"));
  }

  private static void checkPeerAndDefaultPortInConstructorCall(SubscriptionContext ctx, CallExpression callExpression) {
    getArgument(ctx, callExpression, OTHER, 0)
      .filter(flow -> flow.hasExpression(IS_SENSITIVE_PEER))
      .ifPresent(flow -> checkDefaultPortInConstructorCall(ctx, callExpression));
  }

  private static void checkDefaultPortInConstructorCall(SubscriptionContext ctx, CallExpression callExpression) {
    Expression expression = callExpression.callee();
    if (expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      expression = ((QualifiedExpression) expression).qualifier();
    }

    // trace back the creation of the object to check if a sensitive port was specified as default
    CdkUtils.ExpressionFlow flowObj = CdkUtils.ExpressionFlow.build(ctx, expression);
    flowObj.getExpression(isCallExpression().and(isFqn("aws_cdk.aws_ec2.Connections")))
      .map(CallExpression.class::cast)
      .flatMap(callExpr -> getArgument(ctx, callExpr, "default_port"))
      .filter(flow -> flow.hasExpression(IS_SENSITIVE_PORT))
      .ifPresent(flow -> ctx.addIssue(getMethodPrimaryLocation(callExpression), MESSAGE_BAD_METHOD));
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkPeerAndPortSensitivity(String peerName, String portName) {
    return (ctx, callExpression) ->
      getArgument(ctx, callExpression, peerName, 0)
        .filter(flow -> flow.hasExpression(IS_SENSITIVE_PEER))
        .flatMap(flow -> getArgument(ctx, callExpression, portName, 1))
        .filter(flow -> flow.hasExpression(IS_SENSITIVE_PORT))
        .flatMap(flow -> getArgument(ctx, callExpression, peerName, 0))
        .ifPresent(flow -> flow.addIssue(MESSAGE_BAD_PEER));
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkPortSensitivity(String portName) {
    return (ctx, callExpression) ->
      getArgument(ctx, callExpression, portName, 0)
        .filter(flow -> flow.hasExpression(IS_SENSITIVE_PORT))
        .ifPresent(flow -> ctx.addIssue(getMethodPrimaryLocation(callExpression), MESSAGE_BAD_METHOD));
  }

  private static Expression getMethodPrimaryLocation(CallExpression callExpression) {
    Expression expression = callExpression.callee();
    if(expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      return ((QualifiedExpression) expression).name();
    }
    return expression;
  }
}
