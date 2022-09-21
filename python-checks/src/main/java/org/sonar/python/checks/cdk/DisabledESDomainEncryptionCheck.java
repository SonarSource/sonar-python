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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6308")
public class DisabledESDomainEncryptionCheck extends AbstractCdkResourceCheck {
  private static final String OMITTING_MESSAGE = "Omitting %s causes encryption of data at rest to be disabled for this {OpenSearch|Elasticsearch} domain." +
    " Make sure it is safe here.";
  private static final String UNENCRYPTED_MESSAGE = "Make sure that using unencrypted {OpenSearch|Elasticsearch} domains is safe here.";
  private static final String ENABLED = "enabled";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_opensearchservice.Domain", DisabledESDomainEncryptionCheck::checkNormalDomain);
    checkFqn("aws_cdk.aws_opensearchservice.CfnDomain", DisabledESDomainEncryptionCheck::checkCfnDomain);
  }

  private static void checkNormalDomain(SubscriptionContext ctx, CallExpression callExpression) {
    checkDomain(ctx, callExpression, "encryption_at_rest", "aws_cdk.aws_opensearchservice.EncryptionAtRestOptions");
  }

  private static void checkCfnDomain(SubscriptionContext ctx, CallExpression callExpression) {
    checkDomain(ctx, callExpression, "encryption_at_rest_options", "aws_cdk.aws_opensearchservice.CfnDomain.EncryptionAtRestOptionsProperty");
  }

  private static void checkDomain(SubscriptionContext ctx, CallExpression callExpression, String encryptionArgName, String argFqnMethod) {
    getArgument(ctx, callExpression, encryptionArgName).ifPresentOrElse(
      argEncryptedTrace ->
        argEncryptedTrace.addIssueIf(expr -> isMethodCallWithoutEnabledOrSetToFalse(ctx, expr, argFqnMethod)
          || (expr.is(Tree.Kind.DICTIONARY_LITERAL) && isDictionaryEnabledValueFalseOrAbsent(ctx, (DictionaryLiteral) expr))
          , UNENCRYPTED_MESSAGE)
      , () -> ctx.addIssue(callExpression.callee(), String.format(OMITTING_MESSAGE, encryptionArgName))
    );
  }

  private static boolean isMethodCallWithoutEnabledOrSetToFalse(SubscriptionContext ctx, Expression expr, String argFqnMethod) {
    if (!AbstractCdkResourceCheck.isFqnValue(expr, argFqnMethod)) {
      return false;
    }
    if (!expr.is(Tree.Kind.CALL_EXPR)) {
      return true;
    }

    return getArgument(ctx, (CallExpression) expr, ENABLED)
      .filter(argEnabledTrace -> !argEnabledTrace.hasExpression(AbstractCdkResourceCheck::isFalse)).isEmpty();
  }

  private static boolean isDictionaryEnabledValueFalseOrAbsent(SubscriptionContext ctx, DictionaryLiteral dictionaryLiteral) {
    return dictionaryLiteral.elements().stream()
      .filter(element -> element.is(Tree.Kind.KEY_VALUE_PAIR)).map(KeyValuePair.class::cast)
      .filter(keyValuePair -> keyValuePair.key().is(Tree.Kind.STRING_LITERAL)
        && ((StringLiteral) keyValuePair.key()).trimmedQuotesValue().equals(ENABLED))
      .map(keyValuePair -> ArgumentTrace.build(ctx, keyValuePair.value()))
      .allMatch(argTrace -> argTrace.hasExpression(AbstractCdkResourceCheck::isFalse));
  }
}
