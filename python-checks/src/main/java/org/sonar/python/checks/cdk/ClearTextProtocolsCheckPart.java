/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkPredicate.isFalse;
import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isNone;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionaryInList;

public class ClearTextProtocolsCheckPart extends AbstractCdkResourceCheck {

  private static final String LB_MESSAGE = "Make sure that using network protocols without an SSL/TLS underlay is safe here.";
  private static final String ELASTICACHE_MESSAGE = "Make sure that disabling transit encryption is safe here.";
  private static final String KINESIS_MESSAGE = "Make sure that disabling stream encryption is safe here.";

  private static final String OMITTING_MESSAGE = "Omitting `%s` causes %s encryption to be disabled. Make sure it is safe here.";

  private static final String PROTOCOL = "protocol";
  private static final String EXTERNAL_PROTOCOL_SNAKE_CASE = "external_protocol";
  private static final String EXTERNAL_PROTOCOL_CAMEL_CASE = "externalProtocol";
  private static final String LISTENERS = "listeners";

  /**
   * Constant wrapper of sensitive protocols and ports of AWS::ElasticLoadBalancing
   */
  private static class Elb {
    private static final Set<String> SENSITIVE_TRANSPORT_PROTOCOL_FQNS = Set.of(
      prefix("LoadBalancingProtocol.TCP"),
      prefix("LoadBalancingProtocol.HTTP")
    );
    private static final Set<String> SENSITIVE_TRANSPORT_PROTOCOLS = Set.of("http", "tcp");

    static String prefix(String lbName) {
      return "aws_cdk.aws_elasticloadbalancing." + lbName;
    }
  }

  /**
   * Constant wrapper of sensitive protocols and ports of AWS::ElasticLoadBalancingV2
   */
  private static class Elbv2 {
    private static final String SENSITIVE_HTTP_PROTOCOL_FQN = prefix("ApplicationProtocol.HTTP");
    private static final Set<String> SENSITIVE_TRANSPORT_PROTOCOL_FQNS = Set.of(
      prefix("Protocol.TCP"),
      prefix("Protocol.UDP"),
      prefix("Protocol.TCP_UDP")
    );
    private static final Set<String> SENSITIVE_TRANSPORT_PROTOCOLS = Set.of("HTTP", "TCP", "UDP", "TCP_UDP");

    static String prefix(String lbName) {
      return "aws_cdk.aws_elasticloadbalancingv2." + lbName;
    }
  }

  private static class Kinesis {

    private static final String SENSITIVE_STREAM_ENCRYPTION_FQN = prefix("StreamEncryption.UNENCRYPTED");
    static String prefix(String lbName) {
      return "aws_cdk.aws_kinesis." + lbName;
    }
  }

  private static final Set<Integer> HTTP_PROTOCOL_PORTS = Set.of(80, 8080, 8000, 8008);


  @Override
  protected void registerFqnConsumer() {
    // Raise an issue if a `LoadBalancerListener` is instantiated or `add_listener` is called on an `LoadBalancer` object
    // with the `external_protocol` argument set to  `aws_cdk.aws_elasticloadbalancing.LoadBalancingProtocol.TCP`
    // or `aws_cdk.aws_elasticloadbalancing.LoadBalancingProtocol.HTTP`.
    checkFqns(List.of(Elb.prefix("LoadBalancerListener"), Elb.prefix("LoadBalancer.add_listener")), (ctx, call) ->
      getArgument(ctx, call, EXTERNAL_PROTOCOL_SNAKE_CASE).ifPresent(
        protocol -> protocol.addIssueIf(isSensitiveTransportProtocolFqn(Elb.SENSITIVE_TRANSPORT_PROTOCOL_FQNS), LB_MESSAGE)));


    // Raise an issue if LoadBalancer is instantiated with a `listeners` property set to a nonempty sequence
    // that contains a dict with an `external_protocol` entry set to `aws_cdk.aws_elasticloadbalancing.LoadBalancingProtocol.TCP`
    // or `aws_cdk.aws_elasticloadbalancing.LoadBalancingProtocol.HTTP`.
    checkFqn(Elb.prefix("LoadBalancer"), (ctx, call) ->
      getArgument(ctx, call, LISTENERS).flatMap(CdkUtils::getList).ifPresent(
        listeners -> getDictionaryInList(ctx, listeners)
          .forEach(dict -> checkLoadBalancerListenerDict(ctx, dict))));


    // Raise an issue if a CfnLoadBalancer is instantiated with a `listeners` property set to a Sequence
    // that contains a dict with a `protocol` argument set to `http` or `tcp`.
    checkFqn(Elb.prefix("CfnLoadBalancer"), (ctx, call) ->
      getArgument(ctx, call, LISTENERS).flatMap(CdkUtils::getList).ifPresent(
        listeners -> getDictionaryInList(ctx, listeners)
          .forEach(dict -> checkCfnLoadBalancerListenerDict(ctx, dict))));

    // Raise an issue if a CfnLoadBalancer is instantiated with the `protocol` argument set to `http` or `tcp`.
    checkFqn(Elb.prefix("CfnLoadBalancer.ListenersProperty"), (ctx, call) ->
      getArgument(ctx, call, PROTOCOL).ifPresent(
        protocol -> protocol.addIssueIf(isSensitiveTransportProtocol(Elb.SENSITIVE_TRANSPORT_PROTOCOLS), LB_MESSAGE)));


    // Raise an issue if a `ApplicationListener` is instantiated or `add_listener` is called on an `ApplicationLoadBalancer` object
    // with the `protocol` argument set to  `aws_cdk.aws_elasticloadbalancingv2.ApplicationProtocol.HTTP`
    // or if is not set and the `port` argument set to 80,8080,8000, or 8008.
    checkFqns(List.of(Elbv2.prefix("ApplicationListener"), Elbv2.prefix("ApplicationLoadBalancer.add_listener")), (ctx, call) ->
      getArgument(ctx, call, PROTOCOL).ifPresentOrElse(
        protocol -> protocol.addIssueIf(isFqn(Elbv2.SENSITIVE_HTTP_PROTOCOL_FQN), LB_MESSAGE),
        () -> getArgument(ctx, call, "port").ifPresent(
          port -> port.addIssueIf(isSensitiveHttpProtocolPort(), LB_MESSAGE))));


    // Raise an issue if a `NetworkListener` is instantiated or `add_listener` is called on an `NetworkLoadBalancer` object
    // with the `protocol` argument set to `aws_cdk.aws_elasticloadbalancingv2.Protocol.TCP`, `aws_cdk.aws_elasticloadbalancingv2.Protocol.UDP`,
    // or `aws_cdk.aws_elasticloadbalancingv2.Protocol.TCP_UDP` or if is not set and the `certificates` is an empty list or missing.
    checkFqns(List.of(Elbv2.prefix("NetworkListener"), Elbv2.prefix("NetworkLoadBalancer.add_listener")), (ctx, call) ->
      getArgument(ctx, call, PROTOCOL).ifPresentOrElse(
        protocol -> protocol.addIssueIf(isSensitiveTransportProtocolFqn(Elbv2.SENSITIVE_TRANSPORT_PROTOCOL_FQNS), LB_MESSAGE),
        () -> getArgument(ctx, call, "certificates").ifPresentOrElse(
          certificates ->  certificates.addIssueIf(isEmpty(), LB_MESSAGE, call),
          () -> ctx.addIssue(call, LB_MESSAGE))));


    // Raise an issue if a `CfnListener` is instantiated with the `protocol` property set to `HTTP`, `TCP`, `UDP`, or `TCP_UDP`
    checkFqn(Elbv2.prefix("CfnListener"), (ctx, call) ->
      getArgument(ctx, call, PROTOCOL).ifPresent(
        protocol -> protocol.addIssueIf(isSensitiveTransportProtocol(Elbv2.SENSITIVE_TRANSPORT_PROTOCOLS), LB_MESSAGE)));


    // Raise an issue if a `aws_cdk.aws_elasticache.CfnReplicationGroup` is instantiated with the `transit_encryption_enabled`
    // property is missing or set to `False`
    checkFqn("aws_cdk.aws_elasticache.CfnReplicationGroup", (ctx, call) ->
      getArgument(ctx, call, "transit_encryption_enabled").ifPresentOrElse(
        transitEncryption -> transitEncryption.addIssueIf(isFalse(), ELASTICACHE_MESSAGE),
        () -> ctx.addIssue(call.callee(), String.format(OMITTING_MESSAGE, "transit_encryption_enabled", "transit"))));


    // Raise an issue if a `aws_cdk.aws_kinesis.CfnStream` is instantiated with the `stream_encryption`
    // property is missing or set to `Nonce`
    checkFqn(Kinesis.prefix("CfnStream"), (ctx, call) ->
      getArgument(ctx, call, "stream_encryption").ifPresentOrElse(
        streamEncryption -> streamEncryption.addIssueIf(isNone(), KINESIS_MESSAGE),
        () -> ctx.addIssue(call.callee(), String.format(OMITTING_MESSAGE, "stream_encryption", "stream"))));

    // Raise an issue if a `aws_cdk.aws_kinesis.Stream` is instantiated with the `encryption`
    // property is set to `aws_cdk.aws_kinesis.StreamEncryption.UNENCRYPTED`
    checkFqn(Kinesis.prefix("Stream"), (ctx, call) ->
      getArgument(ctx, call, "encryption").ifPresent(
        encryption -> encryption.addIssueIf(isFqn(Kinesis.SENSITIVE_STREAM_ENCRYPTION_FQN), KINESIS_MESSAGE)));
  }

  private static void checkLoadBalancerListenerDict(SubscriptionContext ctx, DictionaryLiteral dict) {
    checkKeyValuePair(ctx, dict, Set.of(EXTERNAL_PROTOCOL_SNAKE_CASE, EXTERNAL_PROTOCOL_CAMEL_CASE), isSensitiveTransportProtocolFqn(Elb.SENSITIVE_TRANSPORT_PROTOCOL_FQNS));
  }

  private static void checkCfnLoadBalancerListenerDict(SubscriptionContext ctx, DictionaryLiteral dict) {
    checkKeyValuePair(ctx, dict, PROTOCOL, isSensitiveTransportProtocol(Elb.SENSITIVE_TRANSPORT_PROTOCOLS));
  }

  private static void checkKeyValuePair(SubscriptionContext ctx, DictionaryLiteral dict, String key, Predicate<Expression> expected) {
    checkKeyValuePair(ctx, dict, Set.of(key), expected);
  }

  private static void checkKeyValuePair(SubscriptionContext ctx, DictionaryLiteral dict, Set<String> keys, Predicate<Expression> expected) {
    keys.stream()
        .map(key -> CdkUtils.getDictionaryPair(ctx, dict, key))
        .filter(Optional::isPresent)
        .findFirst()
        .map(Optional::get)
        .ifPresent(pair -> pair.value.addIssueIf(expected, LB_MESSAGE));
  }

  // ---------------------------------------------------------------------------------------
  // Rule related predicates
  // ---------------------------------------------------------------------------------------

  /**
   * @return Predicate which tests if expression is empty list literal
   */
  private static Predicate<Expression> isEmpty() {
    return expression -> expression.is(Tree.Kind.LIST_LITERAL) && ((ListLiteral) expression).elements().expressions().isEmpty();
  }

  /**
   * @return Predicate which tests if expression is a string and is listed in sensitive transport protocol list
   */
  private static Predicate<Expression> isSensitiveTransportProtocol(Collection<String> transportProtocols) {
    return expression -> CdkUtils.getString(expression).filter(transportProtocols::contains).isPresent();
  }

  /**
   * @return Predicate which tests if expression is a FQN and is listed in sensitive transport protocol FQN list
   */
  private static Predicate<Expression> isSensitiveTransportProtocolFqn(Collection<String> transportProtocolFqns) {
    return expression -> TreeUtils.fullyQualifiedNameFromExpression(expression)
      .filter(transportProtocolFqns::contains).isPresent();
  }

  /**
   * @return Predicate which tests if expression is an integer and is in sensitive port list
   */
  private static Predicate<Expression> isSensitiveHttpProtocolPort() {
    return expression -> CdkUtils.getInt(expression).filter(HTTP_PROTOCOL_PORTS::contains).isPresent();
  }


}
