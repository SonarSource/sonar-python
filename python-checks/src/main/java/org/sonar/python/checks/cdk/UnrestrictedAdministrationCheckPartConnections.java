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

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.checks.cdk.CdkPredicate.hasArgument;
import static org.sonar.python.checks.cdk.CdkPredicate.hasIntervalArguments;
import static org.sonar.python.checks.cdk.CdkPredicate.isCallExpression;
import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isNumeric;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

public class UnrestrictedAdministrationCheckPartConnections extends AbstractCdkResourceCheck {
  private static final String MESSAGE_BAD_PEER = "Change this IP range to a subset of trusted IP addresses.";
  private static final String MESSAGE_BAD_METHOD = "Change this method for `allow_from` and set `other` to a subset of trusted IP addresses.";

  private static final String OTHER = "other";
  private static final String PORT_RANGE = "port_range";
  private static final Set<Long> ADMIN_PORTS = Set.of(22L, 3389L);
  private static final Set<String> CONSTRUCTS_WITH_CONNECTIONS_ATTRIBUTES = Set.of(
    "aws_cdk.aws_docdb.DatabaseCluster", "aws_cdk.aws_lambda_python_alpha.PythonFunction", "aws_cdk.aws_batch_alpha.ComputeEnvironment",
    "aws_cdk.aws_efs.FileSystem", "aws_cdk.aws_lambda_go_alpha.GoFunction", "aws_cdk.aws_ecs.ExternalService", "aws_cdk.aws_ecs.FargateService",
    "aws_cdk.aws_ecs.Cluster", "aws_cdk.aws_ecs.Ec2Service", "aws_cdk.aws_elasticsearch.Domain", "aws_cdk.aws_neptune_alpha.DatabaseCluster",
    "aws_cdk.aws_eks.FargateCluster", "aws_cdk.aws_eks.Cluster", "aws_cdk.aws_codebuild.PipelineProject", "aws_cdk.aws_codebuild.Project",
    "aws_cdk.aws_rds.DatabaseInstance", "aws_cdk.aws_rds.DatabaseInstanceReadReplica", "aws_cdk.aws_rds.DatabaseCluster",
    "aws_cdk.aws_rds.ServerlessClusterFromSnapshot", "aws_cdk.aws_rds.DatabaseProxy", "aws_cdk.aws_rds.DatabaseInstanceFromSnapshot",
    "aws_cdk.aws_rds.ServerlessCluster", "aws_cdk.aws_rds.DatabaseClusterFromSnapshot", "aws_cdk.aws_lambda_nodejs.NodejsFunction",
    "aws_cdk.aws_fsx.LustreFileSystem", "aws_cdk.aws_ec2.BastionHostLinux", "aws_cdk.aws_ec2.ClientVpnEndpoint", "aws_cdk.aws_ec2.Instance",
    "aws_cdk.aws_ec2.LaunchTemplate", "aws_cdk.aws_ec2.SecurityGroup", "aws_cdk.aws_kinesisfirehose_alpha.DeliveryStream",
    "aws_cdk.aws_stepfunctions_tasks.SageMakerCreateTrainingJob", "aws_cdk.aws_stepfunctions_tasks.SageMakerCreateModel",
    "aws_cdk.aws_stepfunctions_tasks.EcsRunTask", "aws_cdk.aws_redshift_alpha.Cluster", "aws_cdk.aws_opensearchservice.Domain",
    "aws_cdk.aws_secretsmanager.HostedRotation", "aws_cdk.aws_msk_alpha.Cluster", "aws_cdk.triggers.TriggerFunction", "aws_cdk.aws_autoscaling.AutoScalingGroup",
    "aws_cdk.aws_synthetics_alpha.Canary", "aws_cdk.aws_cloudfront.experimental.EdgeFunction", "aws_cdk.aws_lambda.Function",
    "aws_cdk.aws_lambda.DockerImageFunction", "aws_cdk.aws_lambda.SingletonFunction", "aws_cdk.aws_lambda.Alias", "aws_cdk.aws_lambda.Version");

  // Predicates to detect sensitive arguments
  private static final Predicate<Expression> IS_SENSITIVE_PROTOCOL =
      isFqn("aws_cdk.aws_ec2.Protocol.ALL")
      .or(isFqn("aws_cdk.aws_ec2.Protocol.TCP"));
  private static final Predicate<Expression> IS_SENSITIVE_PORT =
    isCallExpression().and(
      isFqn("aws_cdk.aws_ec2.Port.all_tcp")
      .or(isFqn("aws_cdk.aws_ec2.Port.all_traffic"))
      .or(isFqn("aws_cdk.aws_ec2.Port.tcp").and(hasArgument("port", 0, isNumeric(ADMIN_PORTS))))
      .or(isFqn("aws_cdk.aws_ec2.Port.tcp_range").and(hasIntervalArguments("start_port", 0, "end_port", 1, ADMIN_PORTS)))
      .or(isFqn("aws_cdk.aws_ec2.Port").and(hasArgument("protocol", IS_SENSITIVE_PROTOCOL)).and(hasIntervalArguments("from_port", "to_port", ADMIN_PORTS)))
    );
  private static final Predicate<Expression> IS_SENSITIVE_PEER =
      isFqn("aws_cdk.aws_ec2.Peer.any_ipv4")
      .or(isFqn("aws_cdk.aws_ec2.Peer.any_ipv6"))
      .or(isFqn("aws_cdk.aws_ec2.Peer.ipv4").and(hasArgument("cidr_ip", 0, isString("0.0.0.0/0"))))
      .or(isFqn("aws_cdk.aws_ec2.Peer.ipv6").and(hasArgument("cidr_ip", 0, isString("::/0"))));

  @Override
  protected void registerFqnConsumer() {
    // Any constructs with 'connections' attributes
    checkFqns(constructsWithPrefix(".connections.allow_from"), checkPeerAndPortSensitivity(OTHER, PORT_RANGE));
    checkFqns(constructsWithPrefix(".connections.allow_from_any_ipv4"), checkPortSensitivity(PORT_RANGE));

    // aws_cdk.aws_ec2.Connections "allow from" methods call
    checkFqn("aws_cdk.aws_ec2.Connections.allow_from", checkPeerAndPortSensitivity(OTHER, PORT_RANGE));
    checkFqn("aws_cdk.aws_ec2.Connections.allow_from_any_ipv4", checkPortSensitivity(PORT_RANGE));
    checkFqn("aws_cdk.aws_ec2.Connections.allow_default_port_from", UnrestrictedAdministrationCheckPartConnections::checkPeerAndDefaultPortInConstructorCall);
    checkFqn("aws_cdk.aws_ec2.Connections.allow_default_port_from_any_ipv4", UnrestrictedAdministrationCheckPartConnections::checkDefaultPortInConstructorCall);

    // SecurityGroup.add_ingress_rule
    checkFqn("aws_cdk.aws_ec2.SecurityGroup.add_ingress_rule", checkPeerAndPortSensitivity("peer", "connection"));
  }

  private static List<String> constructsWithPrefix(String suffix) {
    List<String> result = new ArrayList<>();
    for (String construct : CONSTRUCTS_WITH_CONNECTIONS_ATTRIBUTES) {
      result.add(construct + suffix);
    }
    return result;
  }

  private static void checkPeerAndDefaultPortInConstructorCall(SubscriptionContext ctx, CallExpression callExpression) {
    getArgument(ctx, callExpression, OTHER, 0)
      .filter(flow -> flow.hasExpression(IS_SENSITIVE_PEER))
      .ifPresent(flow -> checkDefaultPortInConstructorCall(ctx, callExpression));
  }

  private static void checkDefaultPortInConstructorCall(SubscriptionContext ctx, CallExpression callExpression) {
    Expression expression = callExpression.callee();
    if (expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      expression = ((QualifiedExpression) expression).qualifier();
    }

    // trace back the creation of the object to check if a sensitive port was specified as default
    CdkUtils.ExpressionFlow flowObj = CdkUtils.ExpressionFlow.build(ctx, expression);
    flowObj.getExpression(isCallExpression().and(isFqn("aws_cdk.aws_ec2.Connections")))
      .map(CallExpression.class::cast)
      .flatMap(callExpr -> getArgument(ctx, callExpr, "default_port"))
      .ifPresent(flow -> {
        if (flow.hasExpression(IS_SENSITIVE_PORT)) {
          ctx.addIssue(getMethodExpression(callExpression), MESSAGE_BAD_METHOD);
        }
      });
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkPeerAndPortSensitivity(String peerName, String portName) {
    return (ctx, callExpression) -> {
      if (getArgument(ctx, callExpression, peerName, 0).filter(flow -> flow.hasExpression(IS_SENSITIVE_PEER)).isPresent()
       && getArgument(ctx, callExpression, portName, 1).filter(flow -> flow.hasExpression(IS_SENSITIVE_PORT)).isPresent()) {
        getArgument(ctx, callExpression, peerName, 0).ifPresent(flow -> flow.addIssue(MESSAGE_BAD_PEER));
      }
    };
  }

  private static BiConsumer<SubscriptionContext, CallExpression> checkPortSensitivity(String portName) {
    return (ctx, callExpression) ->
      getArgument(ctx, callExpression, portName, 0).ifPresent(
        flow -> {
          if(flow.hasExpression(IS_SENSITIVE_PORT)) {
            ctx.addIssue(getMethodExpression(callExpression), MESSAGE_BAD_METHOD);
          }
        });
  }

  private static Expression getMethodExpression(CallExpression callExpression) {
    Expression expression = callExpression.callee();
    if(expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      return ((QualifiedExpression) expression).name();
    }
    return expression;
  }
}
