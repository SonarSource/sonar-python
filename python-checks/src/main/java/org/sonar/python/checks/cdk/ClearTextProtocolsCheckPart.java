/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.cdk;

import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

public class ClearTextProtocolsCheckPart extends AbstractCdkResourceCheck {

  private static final String MESSAGE = "Make sure that using network protocols without an SSL/TLS underlay is safe here.";

  private static final String PROTOCOL = "protocol";
  private static final String EXTERNAL_PROTOCOL = "external_protocol";
  private static final String LISTENERS = "listeners";

  /**
   * Constant wrapper of sensitive protocols and ports of AWS::ElasticLoadBalancing
   */
  private static final class Elb {
    static final Set<String> TRANSPORT_PROTOCOL_FQNS = Set.of(
      prefix("LoadBalancingProtocol.TCP"),
      prefix("LoadBalancingProtocol.HTTP")
    );
    static final Set<String> TRANSPORT_PROTOCOLS = Set.of("http", "tcp");

    static String prefix(String lbName) {
      return "aws_cdk.aws_elasticloadbalancing." + lbName;
    }
  }

  /**
   * Constant wrapper of sensitive protocols and ports of AWS::ElasticLoadBalancingV2
   */
  private static final class Elbv2 {
    static final String HTTP_PROTOCOL_FQN = prefix("ApplicationProtocol.HTTP");
    static final Set<String> TRANSPORT_PROTOCOL_FQNS = Set.of(
      prefix("Protocol.TCP"),
      prefix("Protocol.UDP"),
      prefix("Protocol.TCP_UDP")
    );
    static final Set<String> TRANSPORT_PROTOCOLS = Set.of("HTTP", "TCP", "UDP", "TCP_UDP");

    static String prefix(String lbName) {
      return "aws_cdk.aws_elasticloadbalancingv2." + lbName;
    }
  }

  private static final Set<Integer> HTTP_PROTOCOL_PORTS = Set.of(80, 8080, 8000, 8008);

  /**
   * Register a consumer for multiple FQNs
   */
  private void checkFqns(Collection<String> suffixes, BiConsumer<SubscriptionContext, CallExpression> consumer) {
    suffixes.forEach(suffix -> checkFqn(suffix, consumer));
  }

  @Override
  protected void registerFqnConsumer() {
    // Raise an issue if a `LoadBalancerListener` is instantiated or `add_listener` is called on an `LoadBalancer` object
    // with the `external_protocol` argument set to  `aws_cdk.aws_elasticloadbalancing.LoadBalancingProtocol.TCP`
    // or `aws_cdk.aws_elasticloadbalancing.LoadBalancingProtocol.HTTP`.
    checkFqns(List.of(Elb.prefix("LoadBalancerListener"), Elb.prefix("LoadBalancer.add_listener")), (ctx, call) ->
      getArgument(ctx, call, EXTERNAL_PROTOCOL).ifPresent(
        protocol -> protocol.addIssueIf(isSensitiveTransportProtocolFqn(Elb.TRANSPORT_PROTOCOL_FQNS), MESSAGE)));


    // Raise an issue if LoadBalancer is instantiated with a `listeners` property set to a nonempty sequence
    // that contains a dict with an `external_protocol` entry set to `aws_cdk.aws_elasticloadbalancing.LoadBalancingProtocol.TCP`
    // or `aws_cdk.aws_elasticloadbalancing.LoadBalancingProtocol.HTTP`.
    checkFqn(Elb.prefix("LoadBalancer"), (ctx, call) ->
      getArgumentList(ctx, call, LISTENERS).ifPresent(
        listeners -> getDictionaryInList(ctx, listeners)
          .forEach(dict -> checkLoadBalancerListenerDict(ctx, dict))));


    // Raise an issue if a CfnLoadBalancer is instantiated with a `listeners` property set to a Sequence
    // that contains a dict with a `protocol` argument set to `http` or `tcp`.
    checkFqn(Elb.prefix("CfnLoadBalancer"), (ctx, call) ->
      getArgumentList(ctx, call, LISTENERS).ifPresent(
        listeners -> getDictionaryInList(ctx, listeners)
          .forEach(dict -> checkCfnLoadBalancerListenerDict(ctx, dict))));

    // Raise an issue if a CfnLoadBalancer is instantiated with the `protocol` argument set to `http` or `tcp`.
    checkFqn(Elb.prefix("CfnLoadBalancer.ListenersProperty"), (ctx, call) ->
      getArgument(ctx, call, PROTOCOL).ifPresent(
        protocol -> protocol.addIssueIf(isSensitiveTransportProtocol(Elb.TRANSPORT_PROTOCOLS), MESSAGE)));


    // Raise an issue if a `ApplicationListener` is instantiated or `add_listener` is called on an `ApplicationLoadBalancer` object
    // with the `protocol` argument set to  `aws_cdk.aws_elasticloadbalancingv2.ApplicationProtocol.HTTP`
    // or if is not set and the `port` argument set to 80,8080,8000, or 8008.
    checkFqns(List.of(Elbv2.prefix("ApplicationListener"), Elbv2.prefix("ApplicationLoadBalancer.add_listener")), (ctx, call) ->
      getArgument(ctx, call, PROTOCOL).ifPresentOrElse(
        protocol -> protocol.addIssueIf(isFqn(Elbv2.HTTP_PROTOCOL_FQN), MESSAGE),
        () -> getArgument(ctx, call, "port").ifPresent(
          port -> port.addIssueIf(isHttpProtocolPort(), MESSAGE, call))));


    // Raise an issue if a `NetworkListener` is instantiated or `add_listener` is called on an `NetworkLoadBalancer` object
    // with the `protocol` argument set to `aws_cdk.aws_elasticloadbalancingv2.Protocol.TCP`, `aws_cdk.aws_elasticloadbalancingv2.Protocol.UDP`,
    // or `aws_cdk.aws_elasticloadbalancingv2.Protocol.TCP_UDP` or if is not set and the `certificates` is an empty list or missing.
    checkFqns(List.of(Elbv2.prefix("NetworkListener"), Elbv2.prefix("NetworkLoadBalancer.add_listener")), (ctx, call) ->
      getArgument(ctx, call, PROTOCOL).ifPresentOrElse(
        protocol -> protocol.addIssueIf(isSensitiveTransportProtocolFqn(Elbv2.TRANSPORT_PROTOCOL_FQNS), MESSAGE),
        () -> getArgument(ctx, call, "certificates").ifPresentOrElse(
          certificates ->  certificates.addIssueIf(isEmpty(), MESSAGE, call),
          () -> ctx.addIssue(call, MESSAGE))));


    // Raise an issue if a `CfnListener` is instantiated with the `protocol` property set to `HTTP`, `TCP`, `UDP`, or `TCP_UDP`
    checkFqn(Elbv2.prefix("CfnListener"), (ctx, call) ->
      getArgument(ctx, call, PROTOCOL).ifPresent(
        protocol -> protocol.addIssueIf(isSensitiveTransportProtocol(Elbv2.TRANSPORT_PROTOCOLS), MESSAGE)));
  }

  private static void checkLoadBalancerListenerDict(SubscriptionContext ctx, DictionaryLiteral dict) {
    checkKeyValuePair(ctx, dict, EXTERNAL_PROTOCOL, isSensitiveTransportProtocolFqn(Elb.TRANSPORT_PROTOCOL_FQNS));
  }

  private static void checkCfnLoadBalancerListenerDict(SubscriptionContext ctx, DictionaryLiteral dict) {
    checkKeyValuePair(ctx, dict, PROTOCOL, isSensitiveTransportProtocol(Elb.TRANSPORT_PROTOCOLS));
  }

  private static void checkKeyValuePair(SubscriptionContext ctx, DictionaryLiteral dict, String key, Predicate<Expression> expected) {
    dict.elements().stream().map(ClearTextProtocolsCheckPart::getKeyValuePair).filter(Objects::nonNull)
      .map(pair -> ResolvedKeyValuePair.build(ctx, pair))
      .filter(pair -> pair.key.hasExpression(isStringValue(key)))
      .forEach(pair -> pair.value.addIssueIf(expected, MESSAGE));
  }

  // ---------------------------------------------------------------------------------------
  // General expression utils
  // ---------------------------------------------------------------------------------------

  private static Optional<String> getStringValue(Expression expression) {
    try {
      return Optional.of(((StringLiteral) expression).trimmedQuotesValue());
    } catch (ClassCastException e) {
      return Optional.empty();
    }
  }

  private static Optional<Integer> getIntValue(Expression expression) {
    try {
      return Optional.of((int)((NumericLiteral) expression).valueAsLong());
    } catch (ClassCastException e) {
      return Optional.empty();
    }
  }

  private static Optional<ListLiteral> getArgumentList(SubscriptionContext ctx, CallExpression call, String argumentName) {
    return getArgument(ctx, call, argumentName)
      .flatMap(arg -> arg.getExpression(e -> e.is(Tree.Kind.LIST_LITERAL)))
      .map(ListLiteral.class::cast);

  }

  // ---------------------------------------------------------------------------------------
  // Rule related utils
  // ---------------------------------------------------------------------------------------

  @CheckForNull
  private static KeyValuePair getKeyValuePair(DictionaryLiteralElement element) {
    return element.is(Tree.Kind.KEY_VALUE_PAIR) ? (KeyValuePair) element : null;
  }

  private static List<DictionaryLiteral> getDictionaryInList(SubscriptionContext ctx, ListLiteral listeners) {
    return getListElements(ctx, listeners).stream()
      .map(elm -> elm.getExpression(expr -> expr.is(Tree.Kind.DICTIONARY_LITERAL)))
      .flatMap(Optional::stream)
      .map(DictionaryLiteral.class::cast)
      .collect(Collectors.toList());
  }

  private static List<ExpressionTrace> getListElements(SubscriptionContext ctx, ListLiteral list) {
    return list.elements().expressions().stream()
      .map(expression -> ExpressionTrace.build(ctx, expression))
      .collect(Collectors.toList());
  }

  // ---------------------------------------------------------------------------------------
  // General predicates
  // ---------------------------------------------------------------------------------------

  /**
   * @return Predicate which tests if expression is a string and is equal the expected value
   */
  private static Predicate<Expression> isStringValue(String expectedValue) {
    return expression -> getStringValue(expression).filter(expectedValue::equals).isPresent();
  }

  /**
   * @return Predicate which tests if expression is empty list literal
   */
  private static Predicate<Expression> isEmpty() {
    return expression -> expression.is(Tree.Kind.LIST_LITERAL) && ((ListLiteral) expression).elements().expressions().isEmpty();
  }

  // ---------------------------------------------------------------------------------------
  // Rule related predicates
  // ---------------------------------------------------------------------------------------

  /**
   * @return Predicate which tests if expression is a string and is listed in sensitive transport protocol list
   */
  private static Predicate<Expression> isSensitiveTransportProtocol(Collection<String> transportProtocols) {
    return expression -> getStringValue(expression).filter(transportProtocols::contains).isPresent();
  }

  /**
   * @return Predicate which tests if expression is a FQN and is listed in sensitive transport protocol FQN list
   */
  private static Predicate<Expression> isSensitiveTransportProtocolFqn(Collection<String> transportProtocolFqns) {
    return expression -> Optional.ofNullable(TreeUtils.fullyQualifiedNameFromExpression(expression))
      .filter(transportProtocolFqns::contains).isPresent();
  }

  /**
   * @return Predicate which tests if expression is an integer and is in sensitive port list
   */
  private static Predicate<Expression> isHttpProtocolPort() {
    return expression -> getIntValue(expression).filter(HTTP_PROTOCOL_PORTS::contains).isPresent();
  }

  /**
   * Dataclass to store a resolved KeyValuePair structure
   */
  static class ResolvedKeyValuePair {

    final ExpressionTrace key;
    final ExpressionTrace value;

    private ResolvedKeyValuePair(ExpressionTrace key, ExpressionTrace value) {
      this.key = key;
      this.value = value;
    }

    static ResolvedKeyValuePair build(SubscriptionContext ctx, KeyValuePair pair) {
      return new ResolvedKeyValuePair(ExpressionTrace.build(ctx, pair.key()), ExpressionTrace.build(ctx, pair.value()));
    }
  }
}
