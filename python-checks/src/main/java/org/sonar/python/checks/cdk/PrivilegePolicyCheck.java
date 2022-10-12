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
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;
import org.sonar.python.checks.cdk.CdkUtils.ResolvedKeyValuePair;

@Rule(key = "S6302")
public class PrivilegePolicyCheck extends AbstractCdkResourceCheck {

  private static final String MESSAGE = "Make sure granting all privileges is safe here.";
  private static final String SECONDARY_MESSAGE = "Related effect";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_iam.PolicyStatement", (ctx, call) -> {
      ExpressionFlow effect = CdkUtils.getArgument(ctx, call, "effect").orElse(null);

      if (CdkIamUtils.hasNotAllowEffect(effect)) {
        return;
      }

      CdkUtils.getArgument(ctx, call, "actions")
        .flatMap(resources -> CdkIamUtils.getWildcard(ctx, resources))
        .ifPresent(wildcard -> reportWildcardActionAndEffect(ctx, wildcard, effect));
    });


    checkFqn("aws_cdk.aws_iam.PolicyStatement.from_json", (ctx, call) ->
      CdkIamUtils.getObjectFromJson(ctx, call).ifPresent(json -> checkPolicyStatement(ctx, json)));


    checkFqn("aws_cdk.aws_iam.PolicyDocument.from_json", (ctx, call) ->
      CdkIamUtils.getObjectFromJson(ctx, call).ifPresent(json -> CdkIamUtils.getPolicyStatements(ctx, json)
        .forEach(statement -> checkPolicyStatement(ctx, statement))));
  }

  private static void checkPolicyStatement(SubscriptionContext ctx, DictionaryLiteral statement) {
    List<ResolvedKeyValuePair> pairs = CdkUtils.resolveDictionary(ctx, statement);

    ExpressionFlow effect = CdkUtils.getDictionaryValue(pairs, "Effect").orElse(null);
    if (CdkIamUtils.hasNotAllowEffect(effect)) {
      return;
    }

    CdkUtils.getDictionaryValue(pairs, "Action")
      .flatMap(action -> CdkIamUtils.getWildcard(ctx, action))
      .ifPresent(wildcard -> reportWildcardActionAndEffect(ctx, wildcard, effect));
  }

  private static void reportWildcardActionAndEffect(SubscriptionContext ctx, ExpressionFlow wildcard, ExpressionFlow effect) {
    PreciseIssue issue = ctx.addIssue(wildcard.getLast(), MESSAGE);
    if (effect != null) {
      issue.secondary(effect.asSecondaryLocation(SECONDARY_MESSAGE));
    }
  }
}
