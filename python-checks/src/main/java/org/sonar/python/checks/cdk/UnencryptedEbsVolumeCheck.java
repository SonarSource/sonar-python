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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;

@Rule(key = "S6275")
public class UnencryptedEbsVolumeCheck extends AbstractCdkResourceCheck {

  private static final String PRIMARY_MESSAGE = "Make sure that using unencrypted volumes is safe here.";
  private static final String OMITTING_MESSAGE = "Omitting \"encrypted\" disables volumes encryption. Make sure it is safe here.";

  @Override
  protected String resourceFqn() {
    return "aws_cdk.aws_ec2.Volume";
  }

  @Override
  protected void visitResourceConstructor(SubscriptionContext ctx, CallExpression resourceConstructor) {
    getArgument(ctx, resourceConstructor, "encrypted").ifPresentOrElse(
      argumentTrace -> argumentTrace.addIssueIf(AbstractCdkResourceCheck::isFalse, PRIMARY_MESSAGE),
      () -> ctx.addIssue(resourceConstructor.callee(), OMITTING_MESSAGE)
    );
  }
}
