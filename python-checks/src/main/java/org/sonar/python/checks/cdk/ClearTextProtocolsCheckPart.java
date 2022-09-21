package org.sonar.python.checks.cdk;

import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;

public class ClearTextProtocolsCheckPart extends AbstractCdkResourceCheck {
  @Override
  protected String resourceFqn() {
    return "aws_cdk.aws_elasticloadbalancingv2.ApplicationLoadBalancer.add_listener";
  }

  @Override
  protected void visitResourceConstructor(SubscriptionContext ctx, CallExpression resourceConstructor) {

  }
}
