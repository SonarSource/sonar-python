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

import java.util.Objects;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;

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
    checkFqn(APIGATEWAY_FQN + "CfnDomainName", checkDomainName(isStringValue("TLS_1_0")));

    // OpenSearch & ElasticSearch
    checkFqn(OPENSEARCH_FQN + "Domain", checkDomain(isFqn(OPENSEARCH_FQN + "TLSSecurityPolicy.TLS_1_0")));
    checkFqn(ELASTICSEARCH_FQN + "Domain", checkDomain(isFqn(ELASTICSEARCH_FQN + "TLSSecurityPolicy.TLS_1_0")));
    checkFqn(OPENSEARCH_FQN + "CfnDomain", checkCfnDomain(OPENSEARCH_FQN + "CfnDomain.DomainEndpointOptionsProperty"));
    checkFqn(ELASTICSEARCH_FQN + "CfnDomain", checkCfnDomain(ELASTICSEARCH_FQN + "CfnDomain.DomainEndpointOptionsProperty"));
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkDomainName(Predicate<Expression> predicateIssue) {
    return (ctx, callExpression) -> getArgument(ctx, callExpression, "security_policy").ifPresent(
      argTrace -> argTrace.addIssueIf(predicateIssue, ENFORCE_MESSAGE)
    );
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkDomain(Predicate<Expression> predicateIssue) {
    return (ctx, callExpression) -> getArgument(ctx, callExpression, TLS_SECURITY_POLICY).ifPresentOrElse(
      argTrace -> argTrace.addIssueIf(predicateIssue, ENFORCE_MESSAGE),
      () -> ctx.addIssue(callExpression.callee(), OMITTING_MESSAGE)
    );
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkCfnDomain(String domainOptionName) {
    return (ctx, callExpression) -> getArgument(ctx, callExpression, "domain_endpoint_options").ifPresentOrElse(
      argTrace -> argTrace.addIssueIf(isSensitiveMethod(ctx, domainOptionName, TLS_SECURITY_POLICY, isStringValue(SENSITIVE_TLS_SECURITY_POLICY))
        .or(isSensitiveDictionaryTls(ctx)), ENFORCE_MESSAGE),
      () -> ctx.addIssue(callExpression.callee(), OMITTING_MESSAGE)
    );
  }

  private static Predicate<Expression> isSensitiveDictionaryTls(SubscriptionContext ctx) {
    return expression -> Optional.of(expression)
      .filter(expr -> expr.is(Tree.Kind.DICTIONARY_LITERAL)).map(DictionaryLiteral.class::cast)
      .filter(hasDictionaryKeyValue(ctx, TLS_SECURITY_POLICY, isStringValue(SENSITIVE_TLS_SECURITY_POLICY)))
      .isPresent();
  }

  private static Predicate<DictionaryLiteral> hasDictionaryKeyValue(SubscriptionContext ctx, String key, Predicate<Expression> expected) {
    return dict -> dict.elements().stream().map(ClearTextProtocolsCheckPart::getKeyValuePair).filter(Objects::nonNull)
      .map(pair -> ClearTextProtocolsCheckPart.ResolvedKeyValuePair.build(ctx, pair))
      .filter(pair -> pair.key.hasExpression(isStringValue(key)))
      .allMatch(pair -> pair.value.hasExpression(expected));
  }
}
