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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StringLiteral;

import static org.sonar.python.checks.cdk.CdkPredicate.isListLiteral;
import static org.sonar.python.checks.cdk.CdkPredicate.isNumericLiteral;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkPredicate.isStringLiteral;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getCall;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionary;

@Rule(key="S6321")
public class UnrestrictedAdministrationCheck extends AbstractCdkResourceCheck {
  private static final String MESSAGE = "Change this IP range to a subset of trusted IP addresses.";

  private static final String IP_PROTOCOL = "ip_protocol";
  private static final String CIDR_IP = "cidr_ip";
  private static final String CIDR_IPV6 = "cidr_ipv6";
  private static final String IPPROTOCOL = "ipProtocol";
  private static final String CIDRIP = "cidrIp";
  private static final String CIDRIPV6 = "cidrIpv6";


  private static final Set<String> BAD_PROTOCOL = Set.of("tcp", "6");
  private static final String INVALID_PROTOCOL = "-1";
  private static final String EMPTY_IPV4 = "0.0.0.0/0";
  private static final String EMPTY_IPV6 = "::/0";
  private static final long[] ADMIN_PORTS = new long[]{22, 3389};

  private SubscriptionContext ctx;

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_ec2.CfnSecurityGroup", this::checkCfnSecurityGroup);
    checkFqn("aws_cdk.aws_ec2.CfnSecurityGroupIngress", this::checkCfnSecurityGroupIngress);
  }

  // Checks methods
  private void checkCfnSecurityGroup(SubscriptionContext ctx, CallExpression callExpression) {
    this.ctx = ctx;

    getArgument(ctx, callExpression, "security_group_ingress")
      .flatMap(flow -> flow.getExpression(isListLiteral()).map(ListLiteral.class::cast))
      .ifPresent(list -> list.elements().expressions()
        // Process each expression in the Python list
        .forEach(expression -> {
          // Process each location of those expressions
          CdkUtils.ExpressionFlow flow = CdkUtils.ExpressionFlow.build(ctx, expression);
          for (Expression expr : flow.locations()) {
            raiseIssueIfIngressPropertyCallWithSensitiveArgument(expr);
            raiseIssueIfDictionaryWithSensitiveArgument(expr);
          }
        })
      );
  }

  private void checkCfnSecurityGroupIngress(SubscriptionContext ctx, CallExpression callExpression) {
    this.ctx = ctx;

    if (isBadProtocolWithEmptyIpAddressAndAdminPort(callExpression) || isInvalidProtocolWithEmptyIpAddress(callExpression)) {
      getArgument(ctx, callExpression, CIDR_IP).ifPresent(flow -> flow.addIssue(MESSAGE));
      getArgument(ctx, callExpression, CIDR_IPV6).ifPresent(flow -> flow.addIssue(MESSAGE));
    }
  }

  // Methods to handle call expressions
  private void raiseIssueIfIngressPropertyCallWithSensitiveArgument(Expression expression) {
    getCall(expression, "aws_cdk.aws_ec2.CfnSecurityGroup.IngressProperty")
      .filter(callExpression -> isBadProtocolWithEmptyIpAddressAndAdminPort(callExpression) || isInvalidProtocolWithEmptyIpAddress(callExpression))
      .ifPresent(callExpression -> {
        getArgument(ctx, callExpression, CIDR_IP).ifPresent(flow -> flow.addIssue(MESSAGE));
        getArgument(ctx, callExpression, CIDR_IPV6).ifPresent(flow -> flow.addIssue(MESSAGE));
      });
  }

  private boolean isBadProtocolWithEmptyIpAddressAndAdminPort(CallExpression callExpression) {
    return is(callExpression, IP_PROTOCOL, isString(BAD_PROTOCOL))
      && (is(callExpression, CIDR_IP, isString(EMPTY_IPV4)) || is(callExpression, CIDR_IPV6, isString(EMPTY_IPV6)))
      && rangePortContainAdminPort(callExpression);
  }

  private boolean isInvalidProtocolWithEmptyIpAddress(CallExpression callExpression) {
    return is(callExpression, IP_PROTOCOL, isString(INVALID_PROTOCOL))
      && (is(callExpression, CIDR_IP, isString(EMPTY_IPV4)) || is(callExpression, CIDR_IPV6, isString(EMPTY_IPV6)));
  }

  private boolean rangePortContainAdminPort(CallExpression callExpression) {
    Optional<Long> min = getLongArg(callExpression, "from_port");
    Optional<Long> max = getLongArg(callExpression, "to_port");

    if (min.isEmpty() || max.isEmpty()) {
      return false;
    }

    return isInInterval(min.get(), max.get(), ADMIN_PORTS);
  }

  private Optional<Long> getLongArg(CallExpression callExpression, String name) {
    return getArgument(ctx, callExpression, name)
      .flatMap(flow -> flow.getExpression(isNumericLiteral()))
      .map(NumericLiteral.class::cast)
      .map(NumericLiteral::valueAsLong);
  }

  // Methods to handle dictionaries
  private void raiseIssueIfDictionaryWithSensitiveArgument(Expression expression) {
    getDictionary(expression)
      .ifPresent(dictionary -> {
        DictionaryAsMap map = DictionaryAsMap.build(ctx, dictionary);

        if (isBadProtocolWithEmptyIpAddressAndAdminPort(map) || isInvalidProtocolWithEmptyIpAddress(map)) {
          map.addIssue(CIDRIP, MESSAGE);
          map.addIssue(CIDRIPV6, MESSAGE);
        }
      });
  }

  private boolean isBadProtocolWithEmptyIpAddressAndAdminPort(DictionaryAsMap map) {
    return map.is(IPPROTOCOL, isString(BAD_PROTOCOL))
      && (map.is(CIDRIP, isString(EMPTY_IPV4)) || map.is(CIDRIPV6, isString(EMPTY_IPV6)))
      && rangePortContainAdminPort(map);
  }

  private boolean isInvalidProtocolWithEmptyIpAddress(DictionaryAsMap map) {
    return map.is(IPPROTOCOL, isString(INVALID_PROTOCOL))
      && (map.is(CIDRIP, isString(EMPTY_IPV4)) || map.is(CIDRIPV6, isString(EMPTY_IPV6)));
  }

  private boolean rangePortContainAdminPort(DictionaryAsMap map) {
    Optional<Long> min = getLongArg(map, "fromPort");
    Optional<Long> max = getLongArg(map, "toPort");

    if (min.isEmpty() || max.isEmpty()) {
      return false;
    }

    return isInInterval(min.get(), max.get(), ADMIN_PORTS);
  }

  private Optional<Long> getLongArg(DictionaryAsMap map, String name) {
    return map.get(name, isNumericLiteral())
      .map(NumericLiteral.class::cast)
      .map(NumericLiteral::valueAsLong);
  }

  // Utils methods
  private boolean is(CallExpression callExpression, String name, Predicate<Expression> predicate) {
    return getArgument(ctx, callExpression, name).filter(flow -> flow.hasExpression(predicate)).isPresent();
  }

  private boolean isInInterval(long min, long max, long[] numbers) {
    for (long port : numbers) {
      if (min <= port && port <= max) {
        return true;
      }
    }
    return false;
  }

  // Class to handle Dictionary elements as a Map of ResolvedKeyValuePair, useful in case we need to refer to several elements and not only looking for a specific one.
  // The keys are all resolved String in CdkUtils.ResolvedKeyValuePair.key
  static class DictionaryAsMap {
    Map<String, CdkUtils.ResolvedKeyValuePair> map = new HashMap<>();

    public static DictionaryAsMap build(SubscriptionContext ctx, DictionaryLiteral dictionary) {
      DictionaryAsMap dict = new DictionaryAsMap();

      List<CdkUtils.ResolvedKeyValuePair> pairs = dictionary.elements().stream()
        .map(e -> CdkUtils.getKeyValuePair(ctx, e))
        .filter(Optional::isPresent).map(Optional::get).collect(Collectors.toList());

      for (CdkUtils.ResolvedKeyValuePair pair : pairs) {
        pair.key.getExpression(isStringLiteral()).map(StringLiteral.class::cast)
          .ifPresent(key -> dict.map.put(key.trimmedQuotesValue(), pair));
      }
      return dict;
    }

    public boolean is(String key, Predicate<Expression> valuePredicate) {
      return map.containsKey(key) && map.get(key).value.hasExpression(valuePredicate);
    }

    public Optional<Expression> get(String key, Predicate<Expression> valuePredicate) {
      if (!map.containsKey(key)) {
        return Optional.empty();
      }
      return map.get(key).value.getExpression(valuePredicate);
    }

    public void addIssue(String key, String message) {
      if (map.containsKey(key)) {
        map.get(key).value.addIssue(message);
      }
    }
  }
}
