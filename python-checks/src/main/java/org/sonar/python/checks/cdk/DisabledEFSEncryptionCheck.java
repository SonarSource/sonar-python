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

import org.sonar.check.Rule;

import static org.sonar.python.checks.cdk.CdkPredicate.isFalse;
import static org.sonar.python.checks.cdk.CdkPredicate.isNone;

@Rule(key = "S6332")
public class DisabledEFSEncryptionCheck extends AbstractCdkResourceCheck {

  private static final String MESSAGE = "Make sure that using unencrypted file systems is safe here.";
  private static final String OMITTING_MESSAGE = "Omitting \"encrypted\" disables EFS encryption. Make sure it is safe here.";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_efs.FileSystem", (ctx, fileSystem) ->
      CdkUtils.getArgument(ctx, fileSystem, "encrypted").ifPresent(
        encrypted -> encrypted.addIssueIf(isFalse(), MESSAGE)
      ));

    checkFqn("aws_cdk.aws_efs.CfnFileSystem", (ctx, fileSystem) ->
      CdkUtils.getArgument(ctx, fileSystem, "encrypted").ifPresentOrElse(
        encrypted -> encrypted.addIssueIf(isFalse().or(isNone()), MESSAGE),
        () -> ctx.addIssue(fileSystem.callee(), OMITTING_MESSAGE)
      ));
  }
}
