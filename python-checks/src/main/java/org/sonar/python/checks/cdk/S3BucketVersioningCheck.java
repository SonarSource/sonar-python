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

import java.util.function.BiConsumer;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;

import static org.sonar.python.checks.cdk.CdkPredicate.isFalse;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6252")
public class S3BucketVersioningCheck extends AbstractS3BucketCheck {

  public static final String MESSAGE = "Make sure an unversioned S3 bucket is safe here.";
  public static final String MESSAGE_OMITTING = "Omitting the \"versioned\" argument disables S3 bucket versioning. Make sure it is safe here.";

  @Override
  BiConsumer<SubscriptionContext, CallExpression> visitBucketConstructor() {
    return (ctx, bucket) ->
      getArgument(ctx, bucket, "versioned").ifPresentOrElse(
        version -> version.addIssueIf(isFalse(), MESSAGE),
        () -> ctx.addIssue(bucket.callee(), MESSAGE_OMITTING));
  }
}
