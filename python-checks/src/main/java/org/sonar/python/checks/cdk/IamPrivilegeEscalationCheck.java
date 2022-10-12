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
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.Expression;

import static org.sonar.python.checks.cdk.CdkPredicate.isString;

@Rule(key = "S6317")
public class IamPrivilegeEscalationCheck extends AbstractCdkResourceCheck {

  private static final String ISSUE_MESSAGE_FORMAT = "This policy is vulnerable to the \"%s\" privilege escalation vector. " +
    "Remove permissions or restrict the set of resources they apply to.";

  private static final String SENSITIVE_EFFECT = "aws_cdk.aws_iam.Effect.ALLOW";
  private static final String SENSITIVE_RESOURCE_REGEX = "arn:[^:]*:[^:]*:[^:]*:[^:]*:(role|user|group)/\\*";

  private static final Set<String> SENSITIVE_ACTIONS = Set.of(
    "iam:CreatePolicyVersion",
    "iam:SetDefaultPolicyVersion",
    "iam:CreateAccessKey",
    "iam:CreateLoginProfile",
    "iam:UpdateLoginProfile",
    "iam:AttachUserPolicy",
    "iam:AttachGroupPolicy",
    "iam:AttachRolePolicy",
    "sts:AssumeRole",
    "iam:PutUserPolicy",
    "iam:PutGroupPolicy",
    "iam:PutRolePolicy",
    "iam:AddUserToGroup",
    "iam:UpdateAssumeRolePolicy",
    "iam:PassRole",
    "ec2:RunInstances",
    "lambda:CreateFunction",
    "lambda:InvokeFunction",
    "lambda:AddPermission",
    "lambda:CreateEventSourceMapping",
    "cloudformation:CreateStack",
    "datapipeline:CreatePipeline",
    "datapipeline:PutPipelineDefinition",
    "glue:CreateDevEndpoint",
    "glue:UpdateDevEndpoint",
    "lambda:UpdateFunctionCode"
  );

  private static final Map<Set<String>, String> ATTACK_VECTOR_NAMES = Map.of(
    Set.of("iam:CreatePolicyVersion"), "Create Policy Version",
    Set.of("iam:SetDefaultPolicyVersion"), "Set Default Policy Version",
    Set.of("iam:CreateAccessKey"), "Create AccessKey",
    Set.of("iam:CreateLoginProfile"), "Create Login Profile",
    Set.of("iam:UpdateLoginProfile"), "Update Login Profile ",
    Set.of("iam:AttachUserPolicy"), "Attach User Policy",
    Set.of("iam:AttachGroupPolicy"), "Attach Group Policy",
    Set.of("iam:AttachRolePolicy", "sts:AssumeRole"), "Attach Role Policy"

  );

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_iam.PolicyStatement", (subscriptionContext, callExpression) -> {
      boolean isSensitiveEffect = CdkUtils.getArgument(subscriptionContext, callExpression, "effect")
        .map(expressionFlow -> expressionFlow.hasExpression(CdkPredicate.isFqn(SENSITIVE_EFFECT)))
        .orElse(true);

      Optional<CdkUtils.ExpressionFlow> sensitiveAction = CdkUtils.getArgument(subscriptionContext, callExpression, "actions")
        .flatMap(CdkUtils::getList)
        .flatMap(listLiteral -> CdkUtils.getListElements(subscriptionContext, listLiteral).stream()
          .filter(expressionFlow -> expressionFlow.hasExpression(isString(SENSITIVE_ACTIONS)))
          .findAny());

      Optional<CdkUtils.ExpressionFlow> resources = CdkUtils.getArgument(subscriptionContext, callExpression, "resources")
        .flatMap(CdkUtils::getList)
        .flatMap(listLiteral -> CdkUtils.getListElements(subscriptionContext, listLiteral).stream()
          .filter(expressionFlow -> expressionFlow.hasExpression(isSensitiveResource()))
          .findAny());

      if (isSensitiveEffect && sensitiveAction.isPresent() && resources.isPresent()) {
        resources.get().addIssue(ISSUE_MESSAGE_FORMAT);
      }


    });
  }

  private static Predicate<Expression> isSensitiveResource() {
    return isString("*").or(
      expression -> CdkUtils.getString(expression).filter(string -> string.matches(SENSITIVE_RESOURCE_REGEX)).isPresent());
  }
}
