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

import java.util.Collections;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkPredicate.isCallExpression;
import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isFqnOf;
import static org.sonar.python.checks.cdk.CdkPredicate.isListLiteral;
import static org.sonar.python.checks.cdk.CdkPredicate.isSubscriptionExpression;
import static org.sonar.python.checks.cdk.CdkPredicate.isTrue;
import static org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionary;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionaryPair;

@Rule(key = "S6329")
public class PublicNetworkAccessToCloudResourcesCheck extends AbstractCdkResourceCheck {
  public static final String PUBLICLY_ACCESSIBLE_ARG_NAME = "publicly_accessible";
  private static final String SUBNET_TYPE = "subnet_type";
  private static final String ASSOCIATE_PUBLIC_IP_ADDRESS = "associate_public_ip_address";

  private static final String MESSAGE = "Make sure allowing public network access is safe here.";

  private static final String SENSITIVE_SUBNET = "aws_cdk.aws_ec2.SubnetType.PUBLIC";
  private static final Set<String> SAFE_SUBNET_TYPES = Set.of("ISOLATED", "PRIVATE_ISOLATED", "PRIVATE", "PRIVATE_WITH_NAT");
  private static final Set<String> COMPLIANT_SUBNETS =
    Set.of("aws_cdk.aws_ec2.SubnetType.PRIVATE_ISOLATED", "aws_cdk.aws_ec2.SubnetType.PRIVATE_WITH_EGRESS", "aws_cdk.aws_ec2.SubnetType.PRIVATE_WITH_NAT");

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_dms.CfnReplicationInstance", (subscriptionContext, callExpression) ->
      getArgument(subscriptionContext, callExpression, PUBLICLY_ACCESSIBLE_ARG_NAME).ifPresentOrElse(
        argument -> argument.addIssueIf(isTrue(), MESSAGE),
        () -> subscriptionContext.addIssue(callExpression, MESSAGE)
      )
    );

    checkFqn("aws_cdk.aws_rds.DatabaseInstance", PublicNetworkAccessToCloudResourcesCheck::checkDatabaseInstance);

    checkFqn("aws_cdk.aws_rds.CfnDBInstance", (subscriptionContext, callExpression) ->
      getArgument(subscriptionContext, callExpression, PUBLICLY_ACCESSIBLE_ARG_NAME).ifPresent(
        argument -> argument.addIssueIf(isTrue(), MESSAGE)
      )
    );

    checkFqn("aws_cdk.aws_ec2.Instance", PublicNetworkAccessToCloudResourcesCheck::checkInstance);
    checkFqn("aws_cdk.aws_ec2.CfnInstance", PublicNetworkAccessToCloudResourcesCheck::checkCfnInstance);
  }

  /**
   * Check that a CallExpression (supposedly an aws_cdk.aws_ec2.Instance() call) has a sensitive 'vpc_subnets' argument
   * <pre>aws_cdk.aws_ec2.Instance(vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC))</pre>
   */
  private static void checkInstance(SubscriptionContext ctx, CallExpression callExpression) {
    getArgument(ctx, callExpression, "vpc_subnets")
      .ifPresent(flow -> {
        checkVpcSubnetAsSensitiveSubnetSelectionCall(ctx, flow);
        checkVpcSubnetAsSensitiveDictionary(ctx, flow);
      });
  }

  /**
   * Check that the provided ExpressionFlow lead to a sensitive CallExpression of aws_cdk.aws_ec2.SubnetSelection
   * <pre>aws_cdk.aws_ec2.SubnetSelection(subnet_type=aws_cdk.aws_ec2.SubnetType.PUBLIC)</pre>
   */
  private static void checkVpcSubnetAsSensitiveSubnetSelectionCall(SubscriptionContext ctx, CdkUtils.ExpressionFlow flow) {
    flow.getExpression(isFqn("aws_cdk.aws_ec2.SubnetSelection"))
      .filter(expression -> expression.is(Tree.Kind.CALL_EXPR)).map(CallExpression.class::cast)
      .ifPresent(callExpression ->
        getArgument(ctx, callExpression, SUBNET_TYPE)
          .flatMap(flowArg -> flowArg.getExpression(isFqn(SENSITIVE_SUBNET)))
          .ifPresent(expr -> ctx.addIssue(callExpression.parent(), MESSAGE)));
  }

  /**
   * Check that the provided ExpressionFlow lead to a DictionaryLiteral with a sensitive 'subnet_type' attribute
   * <pre>{"subnet_type" : aws_cdk.aws_ec2.SubnetType.PUBLIC}}</pre>
   */
  private static void checkVpcSubnetAsSensitiveDictionary(SubscriptionContext ctx, CdkUtils.ExpressionFlow flow) {
    getDictionary(flow)
      .flatMap(dictionary -> getDictionaryPair(ctx, dictionary, SUBNET_TYPE))
      .flatMap(element -> element.value.getExpression(isFqn(SENSITIVE_SUBNET)))
      .ifPresent(expression -> raiseIssueOnParent(ctx, expression, Tree.Kind.REGULAR_ARGUMENT));
  }

  /**
   * Check that a CallExpression (supposedly an aws_cdk.aws_ec2.checkCfnInstance() call) has a sensitive 'network_interfaces' argument.
   * Also check if a valid and compliant subnet_id is provided, in which case this CallExpression is not considered as sensitive.
   * <pre>
   *  aws_cdk.aws_ec2.CfnInstance(network_interfaces=[aws_cdk.aws_ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True)]) # Sensitive
   *  aws_cdk.aws_ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : True}]) # Sensitive
   *  aws_cdk.aws_ec2.CfnInstance(network_interfaces=[aws_cdk.aws_ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True,
   *    subnet_id=aws_cdk.aws_ec2.Vpc.select_subnets(subnet_type=aws_cdk.aws_ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0])]) # Compliant
   *  aws_cdk.aws_ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : True,
   *    "subnet_id" : aws_cdk.aws_ec2.Vpc.select_subnets(subnet_type=aws_cdk.aws_ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0]}]) # Compliant
   * </pre>
   */
  private static void checkCfnInstance(SubscriptionContext ctx, CallExpression callExpression) {
    getArgument(ctx, callExpression, "network_interfaces")
      .flatMap(flow -> flow.getExpression(isListLiteral())).map(ListLiteral.class::cast)
      .map(listLiteral -> listLiteral.elements().expressions())
      .orElse(Collections.emptyList())
      .forEach(expression -> {
        checkNetworkInterfacesCallExpression(ctx, expression);
        checkNetworkInterfacesDictionary(ctx, expression);
      });
  }

  /**
   * Check that the provided ExpressionFlow lead to a sensitive CallExpression of aws_cdk.aws_ec2.CfnInstance.NetworkInterfaceProperty
   * <pre>
   *   aws_cdk.aws_ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True) # Sensitive
   *   aws_cdk.aws_ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True,
   *     subnet_id=aws_cdk.aws_ec2.Vpc.select_subnets(subnet_type=aws_cdk.aws_ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0])] # Compliant
   * </pre>
   */
  private static void checkNetworkInterfacesCallExpression(SubscriptionContext ctx, Expression expression) {
    Optional.of(expression)
      .filter(isCallExpression()).map(CallExpression.class::cast)
      .filter(isFqn("aws_cdk.aws_ec2.CfnInstance.NetworkInterfaceProperty"))
      .ifPresent(call -> checkSensitiveOptionWithoutCompliantSubnetDefined(ctx, call));
  }

  private static void checkSensitiveOptionWithoutCompliantSubnetDefined(SubscriptionContext ctx, CallExpression callExpression) {
    Optional<CdkUtils.ExpressionFlow> associatedPublicIpAddress = getArgument(ctx, callExpression, ASSOCIATE_PUBLIC_IP_ADDRESS);
    Optional<CdkUtils.ExpressionFlow> subnetId = getArgument(ctx, callExpression, "subnet_id");

    if (associatedPublicIpAddress.filter(flow -> flow.hasExpression(isTrue())).isPresent()
      && subnetId.filter(PublicNetworkAccessToCloudResourcesCheck::hasPrivateSubnetDefined).isEmpty()) {
      associatedPublicIpAddress.get().addIssue(MESSAGE);
    }
  }

  /**
   * Check that the provided ExpressionFlow lead to a DictionaryLiteral with a sensitive associate_public_ip_address/subnet_id
   * <pre>
   *   {"associate_public_ip_address" : True} # Sensitive
   *   {"associate_public_ip_address" : True, "subnet_id" : ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0]} #Compliant
   * </pre>
   */
  private static void checkNetworkInterfacesDictionary(SubscriptionContext ctx, Expression expression) {
    getDictionary(expression)
      .map(dictionaryLiteral -> UnrestrictedAdministrationCheckPartCfnSecurity.DictionaryAsMap.build(ctx, dictionaryLiteral))
      .ifPresent(dictionaryAsMap -> {
        if (dictionaryAsMap.hasKeyValuePair(ASSOCIATE_PUBLIC_IP_ADDRESS, isTrue())
          && dictionaryAsMap.getValue("subnet_id").filter(PublicNetworkAccessToCloudResourcesCheck::hasPrivateSubnetDefined).isEmpty()) {
          dictionaryAsMap.getKeyString(ASSOCIATE_PUBLIC_IP_ADDRESS).ifPresent(expr -> raiseIssueOnParent(ctx, expr, Tree.Kind.KEY_VALUE_PAIR));
        }
      });
  }

  /**
   * Check that the provided ExpressionFlow has a compliant (private subnet_type) CallExpression over aws_cdk.aws_ec2.Vpc.select_subnets() method.
   * <pre>aws_cdk.aws_ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0]</pre>
   */
  private static boolean hasPrivateSubnetDefined(CdkUtils.ExpressionFlow subnetId) {
    return subnetId.getExpression(isSubscriptionExpression())
      .map(SubscriptionExpression.class::cast).map(SubscriptionExpression::object)
      .filter(PublicNetworkAccessToCloudResourcesCheck::isCompliantSubnet)
      .isPresent();
  }

  private static boolean isCompliantSubnet(Expression expression) {
    Optional<CallExpression> callExpression = getCallSelectSubnets(expression);

    return callExpression
      .flatMap(call -> getArgument(null, call, SUBNET_TYPE))
      .filter(flow -> flow.hasExpression(isFqnOf(COMPLIANT_SUBNETS)))
      .isPresent();
  }

  private static Optional<CallExpression> getCallSelectSubnets(Expression expression) {
    if (expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      Expression qualifier = ((QualifiedExpression) expression).qualifier();
      CdkUtils.ExpressionFlow flow = CdkUtils.ExpressionFlow.build(null, qualifier);
      return flow.getExpression(isCallExpression().and(isFqn("aws_cdk.aws_ec2.Vpc.select_subnets")))
        .map(CallExpression.class::cast);
    }
    if (expression.is(Tree.Kind.NAME)) {
      CdkUtils.ExpressionFlow flow = CdkUtils.ExpressionFlow.build(null, expression);
      return flow.getExpression(isQualifiedExpression()).map(QualifiedExpression.class::cast)
        .map(qualifiedExpression -> CdkUtils.ExpressionFlow.build(null, qualifiedExpression.qualifier()))
        .flatMap(flow2 -> flow2.getExpression(isCallExpression().and(isFqn("aws_cdk.aws_ec2.Vpc.select_subnets"))))
        .map(CallExpression.class::cast);
    }
    return Optional.empty();
  }

  private static void raiseIssueOnParent(SubscriptionContext ctx, Expression expression, Tree.Kind kind) {
    ctx.addIssue(Optional.ofNullable(TreeUtils.firstAncestorOfKind(expression, kind)).orElse(expression), MESSAGE);
  }

  private static void checkDatabaseInstance(SubscriptionContext ctx, CallExpression call) {
    Optional<ExpressionFlow> vpcSubnets = getArgument(ctx, call, "vpc_subnets");

    Optional<ExpressionFlow> subnetType = vpcSubnets
      .flatMap(flow -> flow.getExpression(isCallExpression().and(isFqn("aws_cdk.aws_ec2.SubnetSelection"))))
      .map(CallExpression.class::cast)
      .flatMap(subnetSelection -> getArgument(ctx, subnetSelection, SUBNET_TYPE));

    if (subnetType.filter(isSafeSubnetSelection()).isPresent()) {
      return;
    }

    // Raise issue if
    //  - vpcSubnets is public and publicly_accessible is true
    //  - vpcSubnets is unknown and publicly_accessible is true
    //  - vpcSubnets is public and publicly_accessible is not set
    getArgument(ctx, call, PUBLICLY_ACCESSIBLE_ARG_NAME).ifPresentOrElse(access -> access.addIssueIf(isTrue(), MESSAGE),
      () -> subnetType.filter(isPublicSubnetSelection()).ifPresent(subnets -> subnets.addIssue(MESSAGE)));
  }

  /**
   * The `vpc_subnets` is safe if it is an `SubnetSelection` object with `subnet_type` of type `SubnetType` and not `PUBLIC`
   */
  private static Predicate<ExpressionFlow> isSafeSubnetSelection() {
    return subnetType -> SAFE_SUBNET_TYPES.stream()
      .anyMatch(safeType -> subnetType.hasExpression(isFqn("aws_cdk.aws_ec2.SubnetType." + safeType)));
  }

  private static Predicate<ExpressionFlow> isPublicSubnetSelection() {
    return subnetType -> subnetType.hasExpression(isFqn(SENSITIVE_SUBNET));
  }

  private static Predicate<Expression> isQualifiedExpression() {
    return expression -> expression.is(Tree.Kind.QUALIFIED_EXPR);
  }
}
