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
import java.util.function.BiConsumer;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.checks.cdk.CdkPredicate.isFalse;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6281")
public class S3BucketBlockPublicAccessCheck extends AbstractS3BucketCheck {

  private static final String MESSAGE = "Make sure allowing public ACL/policies to be set is safe here.";
  private static final String OMITTING_MESSAGE = "No Public Access Block configuration prevents public ACL/policies to be set on this S3 bucket. Make sure it is safe here.";

  private static final String BLOCK_PUBLIC_ACCESS_FQN = "aws_cdk.aws_s3.BlockPublicAccess";
  private static final String BLOCK_ACLS_FQN = BLOCK_PUBLIC_ACCESS_FQN + ".BLOCK_ACLS";
  private static final List<String> BLOCK_PUBLIC_ACCESS_ARGUMENTS = List.of(
    "block_public_acls",
    "ignore_public_acls",
    "block_public_policy",
    "restrict_public_buckets");
    
  @Override
  BiConsumer<SubscriptionContext, CallExpression> visitBucketConstructor() {
    return (ctx, bucket) -> {
      Optional<CdkUtils.ExpressionFlow> blockPublicAccess = getArgument(ctx, bucket, "block_public_access");
      if (blockPublicAccess.isPresent()) {
        checkBlockPublicAccess(ctx, blockPublicAccess.get());
      } else {
        ctx.addIssue(bucket.callee(), OMITTING_MESSAGE);
      }
    };
  }

  private static void checkBlockPublicAccess(SubscriptionContext ctx, CdkUtils.ExpressionFlow blockPublicAccess) {
    blockPublicAccess.addIssueIf(S3BucketBlockPublicAccessCheck::blocksAclsOnly, MESSAGE);
    blockPublicAccess.locations().stream().filter(CallExpression.class::isInstance).map(CallExpression.class::cast)
      .filter(S3BucketBlockPublicAccessCheck::isBlockPublicAccessConstructor)
      .findAny()
      .ifPresent(bpaConstructor -> visitBlockPublicAccessConstructor(ctx, bpaConstructor));
  }

  private static void visitBlockPublicAccessConstructor(SubscriptionContext ctx, CallExpression bpaConstructor) {
    BLOCK_PUBLIC_ACCESS_ARGUMENTS.stream()
      .map(args -> getArgument(ctx, bpaConstructor, args))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .toList()
      .forEach(flow -> flow.addIssueIf(isFalse(), MESSAGE));
  }

  private static boolean blocksAclsOnly(Expression expression) {
    if (expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      QualifiedExpression qualifiedExpression = (QualifiedExpression) expression;
      return Optional.ofNullable(qualifiedExpression.symbol())
        .map(Symbol::fullyQualifiedName)
        .filter(BLOCK_ACLS_FQN::equals)
        .isPresent();
    }
    return false;
  }

  private static boolean isBlockPublicAccessConstructor(CallExpression expression) {
    return Optional.ofNullable(expression.calleeSymbol()).map(Symbol::fullyQualifiedName).filter(BLOCK_PUBLIC_ACCESS_FQN::equals).isPresent();
  }
}
