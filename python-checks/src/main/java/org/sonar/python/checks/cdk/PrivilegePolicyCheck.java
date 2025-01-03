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

import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;

import static org.sonar.python.checks.cdk.CdkPredicate.isWildcard;

@Rule(key = "S6302")
public class PrivilegePolicyCheck extends AbstractIamPolicyStatementCheck {

  private static final String MESSAGE = "Make sure granting all privileges is safe here.";
  private static final String SECONDARY_MESSAGE = "Related effect";

  @Override
  protected void checkAllowingPolicyStatement(PolicyStatement policyStatement) {
    CdkUtils.ExpressionFlow action = policyStatement.actions();

    if (action == null) {
      return;
    }

    Optional.ofNullable(getSensitiveExpression(action, isWildcard()))
      .ifPresent(wildcard -> reportWildcardActionAndEffect(wildcard, policyStatement.effect()));
  }

  private static void reportWildcardActionAndEffect(CdkUtils.ExpressionFlow wildcard, @Nullable CdkUtils.ExpressionFlow effect) {
    PreciseIssue issue = wildcard.ctx().addIssue(wildcard.getLast(), MESSAGE);
    if (effect != null) {
      issue.secondary(effect.asSecondaryLocation(SECONDARY_MESSAGE));
    }
  }
}
