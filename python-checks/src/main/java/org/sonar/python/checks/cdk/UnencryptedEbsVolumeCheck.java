/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import org.sonar.check.Rule;

import static org.sonar.python.checks.cdk.CdkPredicate.isFalse;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6275")
public class UnencryptedEbsVolumeCheck extends AbstractCdkResourceCheck {

  private static final String PRIMARY_MESSAGE = "Make sure that using unencrypted volumes is safe here.";
  private static final String OMITTING_MESSAGE = "Omitting \"encrypted\" disables volumes encryption. Make sure it is safe here.";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_ec2.Volume", (ctx, volume) ->
      getArgument(ctx, volume, "encrypted").ifPresentOrElse(
        flow -> flow.addIssueIf(isFalse(), PRIMARY_MESSAGE),
      () -> ctx.addIssue(volume.callee(), OMITTING_MESSAGE)));
  }
}
