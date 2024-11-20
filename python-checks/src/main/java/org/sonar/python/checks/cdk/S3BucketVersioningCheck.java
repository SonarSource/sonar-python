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
