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
import java.util.Optional;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

public class ClearTextProtocolsCheckPart extends AbstractCdkResourceCheck {

  private static final String PROTOCOL_MESSAGE = "Make sure that using network protocols without an SSL/TLS underlay is safe here.";
  private static final String HTTP_PROTOCOL_FQN = lb("ApplicationProtocol.HTTP");
  private static final Set<String> TRANSPORT_PROTOCOL_FQNS = Set.of(
    lb("Protocol.TCP"),
    lb("Protocol.UDP"),
    lb("Protocol.TCP_UDP")
  );

  private static final Set<String> TRANSPORT_PROTOCOLS = Set.of(
    "HTTP", "TCP", "UDP", "TCP_UDP"
  );

  private static final Set<Integer> HTTP_PROTOCOL_PORTS = Set.of(80, 8080, 8000, 8008);

  private void checkFqns(Collection<String> suffixes, BiConsumer<SubscriptionContext, CallExpression> consumer) {
    suffixes.forEach(suffix -> checkFqn(suffix, consumer));
  }

  /**
   * To increase readability and set focus on class and function name, complete FQN of AWS::ElasticLoadBalancingV2
   */
  private static String lb(String lbName) {
    return "aws_cdk.aws_elasticloadbalancingv2." + lbName;
  }

  @Override
  protected void registerFqnConsumer() {
    checkFqns(List.of(lb("ApplicationListener"), lb("ApplicationLoadBalancer.add_listener")),
      this::checkApplicationListener);

    checkFqns(List.of(lb("NetworkListener"), lb("NetworkLoadBalancer.add_listener")),
      this::checkNetworkListener);

    checkFqn(lb("CfnListener"), (ctx, call) ->
      getProtocolArgument(ctx, call).ifPresent(
        protocol -> protocol.addIssueIf(isSensitiveTransportProtocol(), PROTOCOL_MESSAGE)
      ));
  }

  private void checkApplicationListener(SubscriptionContext ctx, CallExpression call) {
    getProtocolArgument(ctx, call).ifPresentOrElse(
      protocol -> protocol.addIssueIf(isFqn(HTTP_PROTOCOL_FQN), PROTOCOL_MESSAGE),
      () -> getArgument(ctx, call, "port").ifPresent(
        port -> port.addIssueIf(isHttpProtocolPort(), PROTOCOL_MESSAGE, call)));
  }

  private void checkNetworkListener(SubscriptionContext ctx, CallExpression call) {
    getProtocolArgument(ctx, call).ifPresentOrElse(
      protocol -> protocol.addIssueIf(isSensitiveTransportProtocolFqn(), PROTOCOL_MESSAGE),
      () -> getArgument(ctx, call, "certificates").ifPresentOrElse(
        certificates ->  certificates.addIssueIf(isEmpty(), PROTOCOL_MESSAGE, call),
        () -> ctx.addIssue(call, PROTOCOL_MESSAGE)
      ));
  }

  private static Optional<ArgumentTrace> getProtocolArgument(SubscriptionContext ctx, CallExpression call) {
    return getArgument(ctx, call, "protocol");
  }

  /**
   * @return Predicate which tests if expression is a string and is listed in sensitive transport protocol list
   */
  private static Predicate<Expression> isSensitiveTransportProtocol() {
    return expression -> getStringValue(expression).filter(TRANSPORT_PROTOCOLS::contains).isPresent();
  }

  /**
   * @return Predicate which tests if expression is a FQN and is listed in sensitive transport protocol FQN list
   */
  private static Predicate<Expression> isSensitiveTransportProtocolFqn() {
    return expression -> Optional.ofNullable(TreeUtils.fullyQualifiedNameFromExpression(expression))
      .filter(TRANSPORT_PROTOCOL_FQNS::contains).isPresent();
  }

  /**
   * @return Predicate which tests if expression is empty list literal
   */
  private static Predicate<Expression> isEmpty() {
    return expression -> expression.is(Tree.Kind.LIST_LITERAL) && ((ListLiteral) expression).elements().expressions().isEmpty();
  }

  /**
   * @return Predicate which tests if expression is an integer and is in sensitive port list
   */
  private static Predicate<Expression> isHttpProtocolPort() {
    return expression -> getIntValue(expression).filter(HTTP_PROTOCOL_PORTS::contains).isPresent();
  }

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
}
