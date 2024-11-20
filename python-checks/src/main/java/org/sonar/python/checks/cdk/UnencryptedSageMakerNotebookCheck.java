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

import static org.sonar.python.checks.cdk.CdkPredicate.isNone;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6319")
public class UnencryptedSageMakerNotebookCheck extends AbstractCdkResourceCheck {
  private static final String OMITTING_MESSAGE = "Omitting kms_key_id disables encryption of SageMaker notebook instances. Make sure it is safe here.";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_sagemaker.CfnNotebookInstance", (ctx, callExpression) ->
      getArgument(ctx, callExpression, "kms_key_id")
        .ifPresentOrElse(
          argKmsKey -> argKmsKey.addIssueIf(isNone(), OMITTING_MESSAGE),
          () -> ctx.addIssue(callExpression.callee(), OMITTING_MESSAGE)
        )
    );
  }
}
