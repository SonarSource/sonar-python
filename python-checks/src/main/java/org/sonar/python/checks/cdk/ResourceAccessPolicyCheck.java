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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;
import org.sonar.python.checks.utils.FileResourceUtils;

import static org.sonar.python.checks.cdk.CdkPredicate.isWildcard;

@Rule(key = "S6304")
public class ResourceAccessPolicyCheck extends AbstractIamPolicyStatementCheck {

  private static final String MESSAGE = "Make sure granting access to all resources is safe here.";
  private static final String SECONDARY_MESSAGE = "Related effect";
  private static final Set<String> SENSITIVE_AWS_ACTIONS;
  private static final String FILEPATH_SENSITIVE_AWS_ACTIONS = "/org/sonar/python/checks/cdk/sensitiveAwsActions.txt";

  static {
    try {
      SENSITIVE_AWS_ACTIONS = FileResourceUtils.loadResourceAsSet(FILEPATH_SENSITIVE_AWS_ACTIONS, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new FileResourceUtils.MissingResourceException(e);
    }
  }

  @Override
  protected void checkAllowingPolicyStatement(PolicyStatement policyStatement) {
    CdkUtils.ExpressionFlow actions = policyStatement.actions();
    CdkUtils.ExpressionFlow resources = policyStatement.resources();

    if (resources == null || actions == null || !isSensitiveAction(actions)) {
      return;
    }

    Optional.ofNullable(getSensitiveExpression(resources, isWildcard()))
      .ifPresent(wildcard -> reportWildcardResourceAndEffect(wildcard, policyStatement.effect()));
  }

  private static boolean isSensitiveAction(ExpressionFlow actions) {
    return getSensitiveExpression(actions, inSensitiveSet()) != null;
  }

  public static Predicate<Expression> inSensitiveSet() {
    return expression -> CdkUtils.getString(expression)
      .filter(SENSITIVE_AWS_ACTIONS::contains).isPresent();
  }

  private static void reportWildcardResourceAndEffect(ExpressionFlow wildcard, @Nullable ExpressionFlow effect) {
    PreciseIssue issue = wildcard.ctx().addIssue(wildcard.getLast(), MESSAGE);
    if (effect != null) {
      issue.secondary(effect.asSecondaryLocation(SECONDARY_MESSAGE));
    }
  }
}
