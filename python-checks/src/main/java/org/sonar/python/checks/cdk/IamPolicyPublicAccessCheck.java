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
import java.util.List;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkPredicate.isFqnOf;
import static org.sonar.python.checks.cdk.CdkPredicate.isWildcard;
import static org.sonar.python.checks.cdk.CdkUtils.getCall;

@Rule(key = "S6270")
public class IamPolicyPublicAccessCheck extends AbstractIamPolicyStatementCheck {

  private static final String ISSUE_MESSAGE = "Make sure granting public access is safe here.";
  private static final String SECONDARY_MESSAGE = "Related effect.";

  @Override
  protected void checkAllowingPolicyStatement(PolicyStatement policyStatement) {
    CdkUtils.ExpressionFlow principals = policyStatement.principals();

    if (principals == null) {
      return;
    }

    CdkUtils.getListExpression(principals)
      .map(list -> CdkUtils.getListElements(principals.ctx(), list))
      .orElse(Collections.emptyList())
      .forEach(principalElement -> raiseIssueIf(principalElement, isSensitivePrincipal(), policyStatement.effect()));
  }

  @Override
  protected void checkPolicyStatementFromJson(PolicyStatement policyStatement) {
    CdkUtils.ExpressionFlow effect = policyStatement.effect();
    CdkUtils.ExpressionFlow principals = policyStatement.principals();

    if (principals == null || !hasAllowEffect(effect)) {
      return;
    }

    raiseIssueIf(principals, isWildcard(), effect);

    CdkUtils.getDictionary(principals)
      .flatMap(innerDict -> CdkUtils.getDictionaryPair(principals.ctx(), innerDict, "AWS"))
      .map(aws -> getSensitiveExpression(aws.value, isWildcard()))
      .ifPresent(sensitiveAwsPrincipal -> raiseIssueIf(sensitiveAwsPrincipal, isWildcard(), effect));
  }

  private static Predicate<Expression> isSensitivePrincipal() {
    return isFqnOf(List.of("aws_cdk.aws_iam.StarPrincipal", "aws_cdk.aws_iam.AnyPrincipal"))
      .or(isSensitiveArnPrincipal());
  }

  private static Predicate<Expression> isSensitiveArnPrincipal() {
    return expression -> getCall(expression, "aws_cdk.aws_iam.ArnPrincipal")
      .map(callExpression -> TreeUtils.nthArgumentOrKeyword(0, "arn", callExpression.arguments()))
      .map(RegularArgument::expression)
      .filter(isWildcard())
      .isPresent();
  }

  private static void raiseIssueIf(CdkUtils.ExpressionFlow expressionFlow, Predicate<Expression> predicate, @Nullable CdkUtils.ExpressionFlow effect) {
    if (effect != null) {
      expressionFlow.addIssueIf(predicate, ISSUE_MESSAGE, effect.asSecondaryLocation(SECONDARY_MESSAGE));
    } else {
      expressionFlow.addIssueIf(predicate, ISSUE_MESSAGE);
    }
  }
}
