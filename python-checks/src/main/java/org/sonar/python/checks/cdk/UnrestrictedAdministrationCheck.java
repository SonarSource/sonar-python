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


  private static final Set<String> SENSITIVE_PROTOCOL = Set.of("tcp", "6");
  private static final String ANY_PROTOCOL = "-1";
  private static final String EMPTY_IPV4 = "0.0.0.0/0";
  private static final String EMPTY_IPV6 = "::/0";
  private static final long[] ADMIN_PORTS = new long[]{22, 3389};

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_ec2.CfnSecurityGroup", UnrestrictedAdministrationCheck::checkCfnSecurityGroup);
    checkFqn("aws_cdk.aws_ec2.CfnSecurityGroupIngress", UnrestrictedAdministrationCheck::checkCfnSecurityGroupIngress);
  }

  // Checks methods
  private static void checkCfnSecurityGroup(SubscriptionContext ctx, CallExpression callExpression) {
    getArgument(ctx, callExpression, "security_group_ingress")
      .flatMap(flow -> flow.getExpression(isListLiteral()).map(ListLiteral.class::cast))
      .ifPresent(list -> list.elements().expressions()
        // Process each expression in the Python list
        .forEach(expression -> {
          CdkUtils.ExpressionFlow flow = CdkUtils.ExpressionFlow.build(ctx, expression);
          raiseIssueIfIngressPropertyCallWithSensitiveArgument(ctx, flow.getLast());
          raiseIssueIfDictionaryWithSensitiveArgument(ctx, flow.getLast());
        })
      );
  }

  private static void checkCfnSecurityGroupIngress(SubscriptionContext ctx, CallExpression callExpression) {
    checkCallCfnSecuritySensitive(new Call(ctx, callExpression));
  }

  // Methods to handle call expressions
  private static void raiseIssueIfIngressPropertyCallWithSensitiveArgument(SubscriptionContext ctx, Expression expression) {
    getCall(expression, "aws_cdk.aws_ec2.CfnSecurityGroup.IngressProperty")
      .map(callExpression -> new Call(ctx, callExpression))
      .ifPresent(UnrestrictedAdministrationCheck::checkCallCfnSecuritySensitive);
  }

  private static void checkCallCfnSecuritySensitive(Call call) {
    if (isCallWithArgumentBadProtocolEmptyIpAddressAdminPort(call) || isCallWithArgumentInvalidProtocolEmptyIpAddress(call)) {
      getArgument(call.ctx, call.callExpression, CIDR_IP).ifPresent(flow -> flow.addIssue(MESSAGE));
      getArgument(call.ctx, call.callExpression, CIDR_IPV6).ifPresent(flow -> flow.addIssue(MESSAGE));
    }
  }

  private static boolean isCallWithArgumentBadProtocolEmptyIpAddressAdminPort(Call call) {
    return call.hasArgument(IP_PROTOCOL, isString(SENSITIVE_PROTOCOL))
      && (call.hasArgument(CIDR_IP, isString(EMPTY_IPV4)) || call.hasArgument(CIDR_IPV6, isString(EMPTY_IPV6)))
      && call.inInterval("from_port", "to_port", ADMIN_PORTS);
  }

  private static boolean isCallWithArgumentInvalidProtocolEmptyIpAddress(Call call) {
    return call.hasArgument(IP_PROTOCOL, isString(ANY_PROTOCOL))
      && (call.hasArgument(CIDR_IP, isString(EMPTY_IPV4)) || call.hasArgument(CIDR_IPV6, isString(EMPTY_IPV6)));
  }

  // Methods to handle dictionaries
  private static void raiseIssueIfDictionaryWithSensitiveArgument(SubscriptionContext ctx, Expression expression) {
    getDictionary(expression)
      .ifPresent(dictionary -> {
        DictionaryAsMap map = DictionaryAsMap.build(ctx, dictionary);

        if (isDictionaryWithAttributeBadProtocolEmptyIpAddressAdminPort(map) || isDictionaryWithAttributeInvalidProtocolEmptyIpAddress(map)) {
          map.addIssue(CIDRIP, MESSAGE);
          map.addIssue(CIDRIPV6, MESSAGE);
        }
      });
  }

  private static boolean isDictionaryWithAttributeBadProtocolEmptyIpAddressAdminPort(DictionaryAsMap map) {
    return map.is(IPPROTOCOL, isString(SENSITIVE_PROTOCOL))
      && (map.is(CIDRIP, isString(EMPTY_IPV4)) || map.is(CIDRIPV6, isString(EMPTY_IPV6)))
      && map.inInterval("fromPort", "toPort", ADMIN_PORTS);
  }

  private static boolean isDictionaryWithAttributeInvalidProtocolEmptyIpAddress(DictionaryAsMap map) {
    return map.is(IPPROTOCOL, isString(ANY_PROTOCOL))
      && (map.is(CIDRIP, isString(EMPTY_IPV4)) || map.is(CIDRIPV6, isString(EMPTY_IPV6)));
  }

  // Class to handle Dictionary elements as a Map of ResolvedKeyValuePair, useful in case we need to refer to several elements and not only looking for a specific one.
  // The keys are all resolved String in CdkUtils.ResolvedKeyValuePair.key
  static class DictionaryAsMap implements CompareInterval {
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

    public Optional<Long> asLong(String name) {
      return get(name, isNumericLiteral())
        .map(NumericLiteral.class::cast)
        .map(NumericLiteral::valueAsLong);
    }
  }

  static class Call implements CompareInterval {
    SubscriptionContext ctx;
    CallExpression callExpression;

    public Call(SubscriptionContext ctx, CallExpression callExpression) {
      this.ctx = ctx;
      this.callExpression = callExpression;
    }

    boolean hasArgument(String name, Predicate<Expression> predicate) {
      return getArgument(ctx, callExpression, name).filter(flow -> flow.hasExpression(predicate)).isPresent();
    }

    public Optional<Long> asLong(String name) {
      return getArgument(ctx, callExpression, name)
        .flatMap(flow -> flow.getExpression(isNumericLiteral()))
        .map(NumericLiteral.class::cast)
        .map(NumericLiteral::valueAsLong);
    }
  }

  // Define the inInterval method used by both Call and DictionaryAsMap classes
  interface CompareInterval {
    Optional<Long> asLong(String name);

    default boolean inInterval(String minName, String maxName, long[] numbers) {
      Optional<Long> min = asLong(minName);
      Optional<Long> max = asLong(maxName);

      if (min.isEmpty() || max.isEmpty()) {
        return false;
      }

      return isInInterval(min.get(), max.get(), numbers);
    }

    private static boolean isInInterval(long min, long max, long[] numbers) {
      for (long port : numbers) {
        if (min <= port && port <= max) {
          return true;
        }
      }
      return false;
    }
  }
}
