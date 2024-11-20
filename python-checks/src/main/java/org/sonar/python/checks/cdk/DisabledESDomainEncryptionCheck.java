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

import java.util.Collections;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.checks.cdk.CdkPredicate.isFalse;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getCall;

@Rule(key = "S6308")
public class DisabledESDomainEncryptionCheck extends AbstractCdkResourceCheck {
  private static final String OMITTING_MESSAGE = "Omitting %s causes encryption of data at rest to be disabled for this %s domain." +
    " Make sure it is safe here.";
  private static final String UNENCRYPTED_MESSAGE = "Make sure that using unencrypted %s domains is safe here.";
  private static final String ENABLED = "enabled";
  private static final String OPENSEARCH = "OpenSearch";
  private static final String ELASTICSEARCH = "Elasticsearch";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_opensearchservice.Domain",
      checkDomain("encryption_at_rest", "aws_cdk.aws_opensearchservice.EncryptionAtRestOptions", OPENSEARCH));

    checkFqn("aws_cdk.aws_opensearchservice.CfnDomain",
      checkDomain("encryption_at_rest_options", "aws_cdk.aws_opensearchservice.CfnDomain.EncryptionAtRestOptionsProperty", OPENSEARCH));

    checkFqn("aws_cdk.aws_elasticsearch.Domain",
      checkDomain("encryption_at_rest", "aws_cdk.aws_elasticsearch.EncryptionAtRestOptions", ELASTICSEARCH));

    checkFqn("aws_cdk.aws_elasticsearch.CfnDomain",
      checkDomain("encryption_at_rest_options", "aws_cdk.aws_elasticsearch.CfnDomain.EncryptionAtRestOptionsProperty", ELASTICSEARCH));
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkDomain(String encryptionArgName, String argFqnMethod, String engine) {
    return (ctx, callExpression) -> getArgument(ctx, callExpression, encryptionArgName)
      .ifPresentOrElse(
        flow -> flow.addIssueIf(isSensitiveOptionObj(ctx, argFqnMethod).or(isSensitiveOptionDict(ctx)), unencryptedMessage(engine)),
        () -> ctx.addIssue(callExpression.callee(), omittingMessage(encryptionArgName, engine))
      );
  }

  private static  String omittingMessage(String encryptionArgName, String engine) {
    return String.format(OMITTING_MESSAGE, encryptionArgName, engine);
  }

  private static String unencryptedMessage(String engine) {
    return String.format(UNENCRYPTED_MESSAGE, engine);
  }

  /**
   * @return Predicate which tests if the expression is the expected object initialization
   * and if the expected argument is set to false or missing
   */
  private static Predicate<Expression> isSensitiveOptionObj(SubscriptionContext ctx, String argFqnMethod) {
    return expression -> getCall(expression, argFqnMethod)
      .map(call -> getArgument(ctx, call, ENABLED)).stream()
      .anyMatch(enabled -> enabled.isEmpty() || enabled.filter(flow -> flow.hasExpression(isFalse())).isPresent());
  }

  // TODO : refactor below to use new available method 'isDictionaryWithKeyValue' in AbstractCdkResourceCheck
  private static Predicate<Expression> isSensitiveOptionDict(SubscriptionContext ctx) {
    return expression -> expression.is(Tree.Kind.DICTIONARY_LITERAL)
      && asDictionaryKeyValuePairs(expression)
      .filter(isKey(ENABLED))
      .allMatch(isValueFalse(ctx));
  }

  private static Stream<KeyValuePair> asDictionaryKeyValuePairs(Expression expression) {
    return asDictionaryElements(expression)
      .filter(element -> element.is(Tree.Kind.KEY_VALUE_PAIR)).map(KeyValuePair.class::cast);
  }

  private static Stream<DictionaryLiteralElement> asDictionaryElements(Expression expression) {
    return Optional.of(expression)
      .filter(expr -> expr.is(Tree.Kind.DICTIONARY_LITERAL)).map(DictionaryLiteral.class::cast)
      .map(DictionaryLiteral::elements)
      .orElse(Collections.emptyList()).stream();
  }

  private static Predicate<KeyValuePair> isKey(String keyName) {
    return keyValuePair -> Optional.of(keyValuePair)
      .map(KeyValuePair::key)
      .filter(key -> key.is(Tree.Kind.STRING_LITERAL)).map(StringLiteral.class::cast)
      .filter(string -> string.trimmedQuotesValue().equals(keyName))
      .isPresent();
  }

  private static Predicate<KeyValuePair> isValueFalse(SubscriptionContext ctx) {
    return keyValuePair -> CdkUtils.ExpressionFlow.build(ctx, keyValuePair.value()).hasExpression(isFalse());
  }
}
