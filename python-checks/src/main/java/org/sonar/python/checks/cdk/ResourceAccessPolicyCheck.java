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

import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;

import static org.sonar.python.checks.cdk.CdkPredicate.isWildcard;

@Rule(key = "S6304")
public class ResourceAccessPolicyCheck extends AbstractCdkResourceCheck {

  private static final String MESSAGE = "Make sure granting access to all resources is safe here.";
  private static final String SECONDARY_MESSAGE = "Related effect";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_iam.PolicyStatement", (ctx, call) ->
      checkPolicyStatement(PolicyStatement.build(ctx, call)));

    checkFqn("aws_cdk.aws_iam.PolicyStatement.from_json", (ctx, call) ->
      CdkIamUtils.getObjectFromJson(ctx, call).ifPresent(json -> checkPolicyStatement(PolicyStatement.build(ctx, json))));

    checkFqn("aws_cdk.aws_iam.PolicyDocument.from_json", (ctx, call) ->
      CdkIamUtils.getObjectFromJson(ctx, call).ifPresent(json -> CdkIamUtils.getPolicyStatements(ctx, json)
        .forEach(statement -> checkPolicyStatement(PolicyStatement.build(ctx, statement)))));
  }

  private static void checkPolicyStatement(PolicyStatement policyStatement) {
    CdkUtils.ExpressionFlow effect = policyStatement.effect();
    CdkUtils.ExpressionFlow actions = policyStatement.actions();
    CdkUtils.ExpressionFlow resources = policyStatement.resources();

    if (resources == null || CdkIamUtils.hasNotAllowEffect(effect) || hasOnlyKmsActions(actions)) {
      return;
    }

    Optional.ofNullable(CdkIamUtils.getSensitiveExpression(resources, isWildcard()))
      .ifPresent(wildcard -> reportWildcardResourceAndEffect(wildcard, effect));
  }

  private static boolean hasOnlyKmsActions(@Nullable ExpressionFlow actions) {
    if (actions == null) {
      return false;
    }

    return CdkUtils.getListElements(actions).stream()
      .allMatch(flow -> flow.hasExpression(CdkPredicate.startsWith("kms:")));
  }

  private static void reportWildcardResourceAndEffect(ExpressionFlow wildcard, ExpressionFlow effect) {
    PreciseIssue issue = wildcard.ctx().addIssue(wildcard.getLast(), MESSAGE);
    if (effect != null) {
      issue.secondary(effect.asSecondaryLocation(SECONDARY_MESSAGE));
    }
  }
}
