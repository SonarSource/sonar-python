/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
