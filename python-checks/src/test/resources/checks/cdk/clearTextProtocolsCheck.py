from aws_cdk import aws_elasticloadbalancingv2 as elbv2


class LoadBalancerStack(Stack):
    def __init__(self, app: App, id: str) -> None:

        lb = elbv2.ApplicationLoadBalancer(
            self,
            "LB",
            vpc=vpc,
            internet_facing=True
        )

        lb.add_listener( # Sensitive
            "Listener-default",
            port=80,
            open=True
        )
