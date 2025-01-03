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

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StringLiteral;

import static org.sonar.python.checks.cdk.CdkPredicate.isNumericLiteral;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkPredicate.isStringLiteral;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getCall;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionary;

public class UnrestrictedAdministrationCheckPartCfnSecurity extends AbstractCdkResourceCheck {
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
    checkFqn("aws_cdk.aws_ec2.CfnSecurityGroup", UnrestrictedAdministrationCheckPartCfnSecurity::checkCfnSecurityGroup);
    checkFqn("aws_cdk.aws_ec2.CfnSecurityGroupIngress", UnrestrictedAdministrationCheckPartCfnSecurity::checkCallCfnSecuritySensitive);
  }

  // Checks methods
  private static void checkCfnSecurityGroup(SubscriptionContext ctx, CallExpression callExpression) {
    getArgument(ctx, callExpression, "security_group_ingress")
      .flatMap(CdkUtils::getListExpression)
      .map(list -> list.elements().expressions())
      .orElse(Collections.emptyList())
      .stream()
      .map(expression -> CdkUtils.ExpressionFlow.build(ctx, expression))
      .forEach(flow -> {
        raiseIssueIfIngressPropertyCallWithSensitiveArgument(ctx, flow.getLast());
        raiseIssueIfDictionaryWithSensitiveArgument(ctx, flow.getLast());
      });
  }

  // Methods to handle call expressions
  private static void raiseIssueIfIngressPropertyCallWithSensitiveArgument(SubscriptionContext ctx, Expression expression) {
    getCall(expression, "aws_cdk.aws_ec2.CfnSecurityGroup.IngressProperty")
      .ifPresent(callExpression -> checkCallCfnSecuritySensitive(ctx, callExpression));
  }

  private static void checkCallCfnSecuritySensitive(SubscriptionContext ctx, CallExpression callExpression) {
    if (isCallWithArgumentBadProtocolEmptyIpAddressAdminPort(ctx, callExpression) || isCallWithArgumentInvalidProtocolEmptyIpAddress(ctx, callExpression)) {
      getArgument(ctx, callExpression, CIDR_IP).ifPresent(flow -> flow.addIssue(MESSAGE));
      getArgument(ctx, callExpression, CIDR_IPV6).ifPresent(flow -> flow.addIssue(MESSAGE));
    }
  }

  private static boolean isCallWithArgumentBadProtocolEmptyIpAddressAdminPort(SubscriptionContext ctx, CallExpression call) {
    return getArgument(ctx, call, IP_PROTOCOL).filter(flow -> flow.hasExpression(isString(SENSITIVE_PROTOCOL))).isPresent()
      && (getArgument(ctx, call, CIDR_IP).filter(flow -> flow.hasExpression(isString(EMPTY_IPV4))).isPresent()
        || getArgument(ctx, call, CIDR_IPV6).filter(flow -> flow.hasExpression(isString(EMPTY_IPV6))).isPresent())
      && hasSensitivePortRange(call, "from_port", "to_port", ADMIN_PORTS);
  }

  private static boolean isCallWithArgumentInvalidProtocolEmptyIpAddress(SubscriptionContext ctx, CallExpression call) {
    return getArgument(ctx, call, IP_PROTOCOL).filter(flow -> flow.hasExpression(isString(ANY_PROTOCOL))).isPresent()
      && (getArgument(ctx, call, CIDR_IP).filter(flow -> flow.hasExpression(isString(EMPTY_IPV4))).isPresent()
      || getArgument(ctx, call, CIDR_IPV6).filter(flow -> flow.hasExpression(isString(EMPTY_IPV6))).isPresent());
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
    return map.hasKeyValuePair(IPPROTOCOL, isString(SENSITIVE_PROTOCOL))
      && (map.hasKeyValuePair(CIDRIP, isString(EMPTY_IPV4)) || map.hasKeyValuePair(CIDRIPV6, isString(EMPTY_IPV6)))
      && map.hasSensitivePortRange("fromPort", "toPort");
  }

  private static boolean isDictionaryWithAttributeInvalidProtocolEmptyIpAddress(DictionaryAsMap map) {
    return map.hasKeyValuePair(IPPROTOCOL, isString(ANY_PROTOCOL))
      && (map.hasKeyValuePair(CIDRIP, isString(EMPTY_IPV4)) || map.hasKeyValuePair(CIDRIPV6, isString(EMPTY_IPV6)));
  }

  // Class to handle Dictionary elements as a Map of ResolvedKeyValuePair, useful in case we need to refer to several elements and not only looking for a specific one.
  // The keys are all resolved String in CdkUtils.ResolvedKeyValuePair.key
  static class DictionaryAsMap {
    Map<String, CdkUtils.ResolvedKeyValuePair> map = new HashMap<>();

    public static DictionaryAsMap build(SubscriptionContext ctx, DictionaryLiteral dictionary) {
      DictionaryAsMap dict = new DictionaryAsMap();

      List<CdkUtils.ResolvedKeyValuePair> pairs = dictionary.elements().stream()
        .map(e -> CdkUtils.getKeyValuePair(ctx, e))
        .filter(Optional::isPresent).map(Optional::get).toList();

      for (CdkUtils.ResolvedKeyValuePair pair : pairs) {
        pair.key.getExpression(isStringLiteral()).map(StringLiteral.class::cast)
          .ifPresent(key -> dict.map.put(key.trimmedQuotesValue(), pair));
      }
      return dict;
    }

    public boolean hasKeyValuePair(String key, Predicate<Expression> valuePredicate) {
      return map.containsKey(key) && map.get(key).value.hasExpression(valuePredicate);
    }

    public Optional<Expression> get(String key, Predicate<Expression> valuePredicate) {
      if (!map.containsKey(key)) {
        return Optional.empty();
      }
      return map.get(key).value.getExpression(valuePredicate);
    }

    public Optional<Expression> getKeyString(String key) {
      return Optional.ofNullable(map.get(key))
        .flatMap(keyValuePair -> keyValuePair.key.getExpression(isStringLiteral()));
    }

    public Optional<CdkUtils.ExpressionFlow> getValue(String key) {
      return Optional.ofNullable(map.get(key)).map(keyValuePair -> keyValuePair.value);
    }

    public void addIssue(String key, String message) {
      if (map.containsKey(key)) {
        map.get(key).value.addIssue(message);
      }
    }

    public Optional<Long> getArgumentAsLong(String name) {
      return get(name, isNumericLiteral())
        .map(NumericLiteral.class::cast)
        .map(NumericLiteral::valueAsLong);
    }

    boolean hasSensitivePortRange(String minName, String maxName) {
      Optional<Long> min = getArgumentAsLong(minName);
      Optional<Long> max = getArgumentAsLong(maxName);

      if (min.isEmpty() || max.isEmpty()) {
        return false;
      }

      return isInInterval(min.get(), max.get(), ADMIN_PORTS);
    }
  }

  // Utils method to work on arguments
  private static Optional<Long> getArgumentAsLong(CallExpression callExpression, String name) {
    return getArgument(null, callExpression, name)
      .flatMap(flow -> flow.getExpression(isNumericLiteral()))
      .map(NumericLiteral.class::cast)
      .map(NumericLiteral::valueAsLong);
  }

  private static boolean hasSensitivePortRange(CallExpression callExpression, String minName, String maxName, long[] numbers) {
    Optional<Long> min = getArgumentAsLong(callExpression, minName);
    Optional<Long> max = getArgumentAsLong(callExpression, maxName);

    if (min.isEmpty() || max.isEmpty()) {
      return false;
    }

    return isInInterval(min.get(), max.get(), numbers);
  }

  public static boolean isInInterval(long min, long max, long[] numbers) {
    for (long port : numbers) {
      if (min <= port && port <= max) {
        return true;
      }
    }
    return false;
  }
}
