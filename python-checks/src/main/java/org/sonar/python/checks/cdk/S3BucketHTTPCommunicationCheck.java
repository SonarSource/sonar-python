/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.Optional;
import java.util.function.BiConsumer;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6249")
public class S3BucketHTTPCommunicationCheck extends AbstractS3BucketCheck {

  private static final String MESSAGE_NO_POLICY = "No bucket policy enforces HTTPS-only access to this bucket. Make sure it is safe here.";
  private static final String MESSAGE_HTTP_ALLOWED = "Make sure authorizing HTTP requests is safe here.";
  private static final String POLICY_STATEMENT_FQN = "aws_cdk.aws_iam.PolicyStatement";

  @Override
  BiConsumer<SubscriptionContext, CallExpression> visitBucketConstructor() {
    return S3BucketHTTPCommunicationCheck::checkBucketConstructor;
  }

  private static void checkBucketConstructor(SubscriptionContext ctx, CallExpression bucketConstructorCall) {

    var enforceSslArg = CdkUtils.getArgument(ctx, bucketConstructorCall, "enforce_ssl");

    if (enforceSslArg.isEmpty()) {
      // No enforce_ssl parameter specified
      // check if this bucket has add_to_resource_policy calls and a correct policy statement
      getBucketPolicy(bucketConstructorCall)
        .flatMap(S3BucketHTTPCommunicationCheck::getPolicyStatementConstructor)
        .ifPresentOrElse(policyStatementConstructorCall -> visitPolicyStatement(ctx, policyStatementConstructorCall),
          () -> ctx.addIssue(bucketConstructorCall.callee(), MESSAGE_NO_POLICY));
    } else {
      enforceSslArg.get().addIssueIf(CdkPredicate.isFalse(), MESSAGE_HTTP_ALLOWED, bucketConstructorCall);
    }
  }

  private static Optional<CallExpression> getBucketPolicy(CallExpression bucketConstructor) {
    return Optional.ofNullable(TreeUtils.firstAncestorOfKind(bucketConstructor, Tree.Kind.ASSIGNMENT_STMT))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(AssignmentStatement.class))
      .flatMap(S3BucketHTTPCommunicationCheck::getFirstAndOnlyAssignedVariable)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .flatMap(S3BucketHTTPCommunicationCheck::getAddToResourcePolicyCall);
  }

  private static Optional<Expression> getFirstAndOnlyAssignedVariable(AssignmentStatement assignmentStatement) {
    return assignmentStatement.lhsExpressions().stream()
      .filter(lhs -> lhs.expressions().size() == 1)
      .map(lhs -> lhs.expressions().get(0))
      .findFirst();
  }

  private static Optional<CallExpression> getAddToResourcePolicyCall(Name variableName) {
    var symbol = TreeUtils.getSymbolFromTree(variableName);
    if (symbol.isEmpty()) {
      return Optional.empty();
    }

    // Check if the usage is part of a qualified expression like "bucket.add_to_resource_policy"
    return symbol.get().usages().stream()
      .filter(usage -> usage.kind() != Usage.Kind.ASSIGNMENT_LHS)
      .map(usage -> usage.tree().parent())
      .flatMap(TreeUtils.toStreamInstanceOfMapper(QualifiedExpression.class))
      .filter(qualifiedExpr -> CdkPredicate.isFqn("add_to_resource_policy").test(qualifiedExpr.name()))
      .map(QualifiedExpression::parent)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(CallExpression.class))
      .findFirst();
  }

  private static Optional<CallExpression> getPolicyStatementConstructor(CallExpression addResourceToPolicyCall) {
    if (!addResourceToPolicyCall.arguments().isEmpty()) {
      return Optional.of(addResourceToPolicyCall.arguments().get(0))
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
        .map(RegularArgument::expression)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
        .filter(CdkPredicate.isFqn(POLICY_STATEMENT_FQN)::test);
    }
    return Optional.empty();
  }

  private static void visitPolicyStatement(SubscriptionContext ctx, CallExpression policyStatementConstructorCall) {
    if (!isPolicyCompliantForHttpsDeny(ctx, policyStatementConstructorCall)) {
      ctx.addIssue(policyStatementConstructorCall.callee(), MESSAGE_HTTP_ALLOWED);
    }
  }

  private static boolean isPolicyCompliantForHttpsDeny(SubscriptionContext ctx, CallExpression policyStatementConstructorCall) {
    // A compliant policy should have at least the following keyword arguments:
    // 1. effect=iam.Effect.DENY
    // 2. resources=["*"] (wildcard)
    // 3. actions=["s3:*"] (wildcard)
    // 4. principals=["*"] (wildcard)
    // 5. conditions=["SecureTransport:False"] 

    boolean hasDenyEffect = hasArgumentWithFqn(ctx, policyStatementConstructorCall, "effect", "aws_cdk.aws_iam.Effect.DENY");
    boolean hasWildcardResources = argumentListContainsExpectedValue(ctx, policyStatementConstructorCall, "resources", "*");
    boolean hasWildcardActions = argumentListContainsExpectedValue(ctx, policyStatementConstructorCall, "actions", "s3:*");
    boolean hasWildcardPrincipals = argumentListContainsExpectedValue(ctx, policyStatementConstructorCall, "principals", "*");
    boolean hasSecureTransportCondition = argumentListContainsExpectedValue(ctx, policyStatementConstructorCall, "conditions", "SecureTransport:False");

    return hasDenyEffect && hasWildcardResources && hasWildcardActions && hasWildcardPrincipals && hasSecureTransportCondition;
  }

  private static boolean hasArgumentWithFqn(SubscriptionContext ctx, CallExpression call, String argName, String expectedFqn) {
    return CdkUtils.getArgument(ctx, call, argName)
      .map(flow -> flow.hasExpression(CdkPredicate.isFqn(expectedFqn)))
      .orElse(false);
  }

  private static boolean argumentListContainsExpectedValue(SubscriptionContext ctx, CallExpression call, String argName, String expectedValue) {
    return CdkUtils.getArgument(ctx, call, argName)
      .flatMap(flow -> flow.getExpression(e -> e.is(Tree.Kind.LIST_LITERAL)))
      .map(ListLiteral.class::cast)
      .map(list -> list.elements().expressions().stream()
        .anyMatch(expr -> CdkUtils.getString(expr).filter(expectedValue::equals).isPresent()))
      .orElse(false);
  }
}
