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

import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;

import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkPredicate.isStringLiteral;
import static org.sonar.python.checks.cdk.CdkPredicate.matches;

@Rule(key = "S6317")
public class IamPrivilegeEscalationCheck extends AbstractIamPolicyStatementCheck {

  private static final String ISSUE_MESSAGE_FORMAT = "This policy is vulnerable to the \"%s\" privilege escalation vector. " +
    "Remove permissions or restrict the set of resources they apply to.";
  private static final String SECONDARY_MESSAGE = "Permissions are granted on all resources.";
  private static final Pattern SENSITIVE_RESOURCE_PATTERN = Pattern.compile("(^\\*$)|(arn:.*:(role|user|group)/\\*)");

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

  private static final Map<String, String> ATTACK_VECTOR_NAMES = Map.of(
    "iam:CreatePolicyVersion", "Create Policy Version",
    "iam:SetDefaultPolicyVersion", "Set Default Policy Version",
    "iam:CreateAccessKey", "Create Access Key",
    "iam:CreateLoginProfile", "Create Login Profile",
    "iam:UpdateLoginProfile", "Update Login Profile ",
    "iam:AttachUserPolicy", "Attach User Policy",
    "iam:AttachGroupPolicy", "Attach Group Policy",
    "iam:AttachRolePolicy", "Attach Role Policy",
    "sts:AssumeRole", "Attach Role Policy"
  );

  @Override
  protected void checkAllowingPolicyStatement(PolicyStatement policyStatement) {
    ExpressionFlow actions = policyStatement.actions();
    ExpressionFlow resources = policyStatement.resources();

    if (actions == null
      || resources == null
      || policyStatement.principals() != null
      || policyStatement.conditions() != null
    ) {
      return;
    }

    ExpressionFlow sensitiveAction = getSensitiveExpression(actions, isString(SENSITIVE_ACTIONS));
    ExpressionFlow sensitiveResource = getSensitiveExpression(resources, matches(SENSITIVE_RESOURCE_PATTERN));

    if (sensitiveAction != null && sensitiveResource != null) {
      reportSensitiveActionAndResource(sensitiveAction, sensitiveResource);
    }
  }

  private static Optional<String> getAttackVectorName(ExpressionFlow sensitiveAction) {
    return sensitiveAction.getExpression(isStringLiteral())
      .map(StringLiteral.class::cast)
      .map(StringLiteral::trimmedQuotesValue)
      .map(action -> ATTACK_VECTOR_NAMES.getOrDefault(action, null));
  }

  private static void reportSensitiveActionAndResource(ExpressionFlow sensitiveAction, ExpressionFlow resources) {
    String attackVectorName =  getAttackVectorName(sensitiveAction).orElse("");
    String message = String.format(ISSUE_MESSAGE_FORMAT, attackVectorName);
    IssueLocation secondary = sensitiveAction.asSecondaryLocation(SECONDARY_MESSAGE);
    resources.addIssue(message, secondary);
  }
}
