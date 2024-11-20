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

import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getCall;

public class WeakSSLProtocolCheckPart extends AbstractCdkResourceCheck {
  private static final String ENFORCE_MESSAGE = "Change this code to enforce TLS 1.2 or above.";
  private static final String OMITTING_MESSAGE = "Omitting \"tls_security_policy\" enables a deprecated version of TLS. Set it to enforce TLS 1.2 or above.";

  // api gateway
  private static final String APIGATEWAY_FQN = "aws_cdk.aws_apigateway.";
  private static final String APIGATEWAYV2_FQN = "aws_cdk.aws_apigatewayv2.";

  // OpenSearch & ElasticSearch
  private static final String OPENSEARCH_FQN = "aws_cdk.aws_opensearchservice.";
  private static final String ELASTICSEARCH_FQN = "aws_cdk.aws_elasticsearch.";
  private static final String TLS_SECURITY_POLICY = "tls_security_policy";
  private static final String SENSITIVE_TLS_SECURITY_POLICY = "Policy-Min-TLS-1-0-2019-07";

  @Override
  protected void registerFqnConsumer() {
    // Api gateway
    checkFqn(APIGATEWAY_FQN + "DomainName", checkDomainName(isFqn(APIGATEWAY_FQN + "SecurityPolicy.TLS_1_0")));
    checkFqn(APIGATEWAYV2_FQN + "DomainName", checkDomainName(isFqn(APIGATEWAYV2_FQN + "SecurityPolicy.TLS_1_0")));
    checkFqn(APIGATEWAY_FQN + "CfnDomainName", checkDomainName(isString("TLS_1_0")));

    // OpenSearch & ElasticSearch
    checkFqn(OPENSEARCH_FQN + "Domain", checkDomain(isFqn(OPENSEARCH_FQN + "TLSSecurityPolicy.TLS_1_0")));
    checkFqn(ELASTICSEARCH_FQN + "Domain", checkDomain(isFqn(ELASTICSEARCH_FQN + "TLSSecurityPolicy.TLS_1_0")));
    checkFqn(OPENSEARCH_FQN + "CfnDomain", checkCfnDomain(OPENSEARCH_FQN + "CfnDomain.DomainEndpointOptionsProperty"));
    checkFqn(ELASTICSEARCH_FQN + "CfnDomain", checkCfnDomain(ELASTICSEARCH_FQN + "CfnDomain.DomainEndpointOptionsProperty"));
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkDomainName(Predicate<Expression> predicateIssue) {
    return (ctx, callExpression) -> CdkUtils.getArgument(ctx, callExpression, "security_policy").ifPresent(
      flow -> flow.addIssueIf(predicateIssue, ENFORCE_MESSAGE)
    );
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkDomain(Predicate<Expression> predicateIssue) {
    return (ctx, callExpression) -> CdkUtils.getArgument(ctx, callExpression, TLS_SECURITY_POLICY).ifPresentOrElse(
      flow -> flow.addIssueIf(predicateIssue, ENFORCE_MESSAGE),
      () -> ctx.addIssue(callExpression.callee(), OMITTING_MESSAGE)
    );
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkCfnDomain(String domainOptionName) {
    return (ctx, callExpression) -> CdkUtils.getArgument(ctx, callExpression, "domain_endpoint_options").ifPresentOrElse(
      flow -> flow.addIssueIf(isSensitiveOptionObj(ctx, domainOptionName)
        .or(isSensitiveDictionaryTls(ctx)), ENFORCE_MESSAGE),
      () -> ctx.addIssue(callExpression.callee(), OMITTING_MESSAGE)
    );
  }

  /**
   * @return Predicate which tests if the expression is the expected object initialization
   * and if the expected argument is set to a sensitive policy or missing
   */
  private static Predicate<Expression> isSensitiveOptionObj(SubscriptionContext ctx, String fqn) {
    return expression -> getCall(expression, fqn)
      .map(call -> getArgument(ctx, call, TLS_SECURITY_POLICY)).stream()
      .anyMatch(policy -> policy.isEmpty() || policy.filter(flow -> flow.hasExpression(isString(SENSITIVE_TLS_SECURITY_POLICY))).isPresent());
  }

  private static Predicate<Expression> isSensitiveDictionaryTls(SubscriptionContext ctx) {
    return expression -> Optional.of(expression)
      .filter(expr -> expr.is(Tree.Kind.DICTIONARY_LITERAL)).map(DictionaryLiteral.class::cast)
      .filter(hasDictionaryKeyValue(ctx, TLS_SECURITY_POLICY, isString(SENSITIVE_TLS_SECURITY_POLICY)))
      .isPresent();
  }

  private static Predicate<DictionaryLiteral> hasDictionaryKeyValue(SubscriptionContext ctx, String key, Predicate<Expression> expected) {
    return dict -> dict.elements().stream()
      .map(e -> CdkUtils.getKeyValuePair(ctx, e))
      .flatMap(Optional::stream)
      .filter(pair -> pair.key.hasExpression(isString(key)))
      .allMatch(pair -> pair.value.hasExpression(expected));
  }
}
