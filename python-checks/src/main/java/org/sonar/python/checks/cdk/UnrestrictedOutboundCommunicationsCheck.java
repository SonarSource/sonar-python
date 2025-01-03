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

import java.util.List;
import java.util.Optional;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;

import static org.sonar.python.checks.cdk.CdkPredicate.isTrue;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6463")
public class UnrestrictedOutboundCommunicationsCheck extends AbstractCdkResourceCheck {

  public static final String OMITTING_MESSAGE = "Omitting \"allow_all_outbound\" enables unrestricted outbound communications. Make sure it is safe here.";
  public static final String UNRESTRICTED_MESSAGE = "Make sure that allowing unrestricted outbound communications is safe here.";

  private static final String SECURITY_GROUP_FQN = "aws_cdk.aws_ec2.SecurityGroup";

  @Override
  protected void registerFqnConsumer() {
    checkFqns(List.of(SECURITY_GROUP_FQN, "aws_cdk.aws_ec2.SecurityGroup.from_security_group_id"), (subscriptionContext, callExpression) ->
      getArgument(subscriptionContext, callExpression, "allow_all_outbound").ifPresentOrElse(
        argument -> argument.addIssueIf(isTrue(), UNRESTRICTED_MESSAGE),
        () -> raiseIssue(subscriptionContext, callExpression) 
      )
    );
  }

  private static void raiseIssue(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(fqn -> fqn.equals(SECURITY_GROUP_FQN))
      .ifPresent(s -> subscriptionContext.addIssue(callExpression.callee(), OMITTING_MESSAGE));
  }

}
