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
