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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;

import static org.sonar.python.checks.cdk.CdkPredicate.isWildcard;

@Rule(key = "S6304")
public class ResourceAccessPolicyCheck extends AbstractCdkResourceCheck {

  private static final String MESSAGE = "Make sure granting access to all resources is safe here.";
  private static final String SECONDARY_MESSAGE = "Related effect";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_iam.PolicyStatement", (ctx, call) -> {
      ExpressionFlow effect = CdkUtils.getArgument(ctx, call, "effect").orElse(null);
      if (hasOnlyKmsActions(ctx, call) || CdkIamUtils.hasNotAllowEffect(effect)) {
        return;
      }

      CdkUtils.getArgument(ctx, call, "resources")
        .map(resources -> CdkIamUtils.getSensitiveExpression(resources, isWildcard()))
        .ifPresent(wildcard -> reportWildcardResourceAndEffect(ctx, wildcard, effect));
    });

    checkFqn("aws_cdk.aws_iam.PolicyStatement.from_json", (ctx, call) ->
      CdkIamUtils.getObjectFromJson(ctx, call).ifPresent(statement -> checkPolicyStatement(ctx, statement)));

    checkFqn("aws_cdk.aws_iam.PolicyDocument.from_json", (ctx, call) ->
      CdkIamUtils.getObjectFromJson(ctx, call).ifPresent(json -> CdkIamUtils.getPolicyStatements(ctx, json)
        .forEach(statement -> checkPolicyStatement(ctx, statement))));
  }

  private static void checkPolicyStatement(SubscriptionContext ctx, DictionaryLiteral statement) {
    List<CdkUtils.ResolvedKeyValuePair> pairs = CdkUtils.resolveDictionary(ctx, statement);
    ExpressionFlow effect = CdkUtils.getDictionaryValue(pairs, "Effect").orElse(null);
    if (hasOnlyKmsActions(ctx, pairs) || CdkIamUtils.hasNotAllowEffect(effect)) {
      return;
    }

    CdkUtils.getDictionaryValue(pairs, "Resource")
      .map(resource -> CdkIamUtils.getSensitiveExpression(resource, isWildcard()))
      .ifPresent(wildcard -> reportWildcardResourceAndEffect(ctx, wildcard, effect));
  }


  private static boolean hasOnlyKmsActions(SubscriptionContext ctx, CallExpression call) {
    return CdkUtils.getArgument(ctx, call, "actions").flatMap(CdkUtils::getList)
      .filter(actions -> hasOnlyKmsActions(ctx, actions))
      .isPresent();
  }

  private static boolean hasOnlyKmsActions(SubscriptionContext ctx, List<CdkUtils.ResolvedKeyValuePair> json) {
    return CdkUtils.getDictionaryValue(json, "Action")
      .flatMap(CdkUtils::getList)
      .filter(actions -> hasOnlyKmsActions(ctx, actions))
      .isPresent();
  }

  private static boolean hasOnlyKmsActions(SubscriptionContext ctx, ListLiteral actions) {
    return CdkUtils.getListElements(ctx, actions)
      .stream()
      .allMatch(flow -> flow.hasExpression(CdkPredicate.startsWith("kms:")));
  }

  private static void reportWildcardResourceAndEffect(SubscriptionContext ctx, ExpressionFlow wildcard, ExpressionFlow effect) {
    PreciseIssue issue = ctx.addIssue(wildcard.getLast(), MESSAGE);
    if (effect != null) {
      issue.secondary(effect.asSecondaryLocation(SECONDARY_MESSAGE));
    }
  }
}
