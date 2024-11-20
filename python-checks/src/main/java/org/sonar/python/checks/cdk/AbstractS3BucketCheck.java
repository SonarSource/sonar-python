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
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;

public abstract class AbstractS3BucketCheck extends AbstractCdkResourceCheck {

  protected static final String S3_BUCKET_FQN = "aws_cdk.aws_s3.Bucket";

  @Override
  protected void registerFqnConsumer() {
    checkFqn(S3_BUCKET_FQN, this.visitBucketConstructor());
  }

  abstract BiConsumer<SubscriptionContext, CallExpression> visitBucketConstructor();
}
