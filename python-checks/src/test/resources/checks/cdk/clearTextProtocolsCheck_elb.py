import aws_cdk.aws_elasticloadbalancing as elb


class LoadBalancerListenerStack(Stack):
    def __init__(self, app: App, id: str) -> None:
        # Noncompliant@+1 {{Make sure that using network protocols without an SSL/TLS underlay is safe here.}}
        elb.LoadBalancerListener(external_protocol=elb.LoadBalancingProtocol.TCP)
        elb.LoadBalancerListener(external_protocol=elb.LoadBalancingProtocol.HTTP)  # Noncompliant
        elb.LoadBalancerListener(external_protocol=elb.LoadBalancingProtocol.HTTPS)
        elb.LoadBalancerListener(external_protocol=external_protocol)
        elb.LoadBalancerListener()

        lb = elb.LoadBalancer()
        lb.add_listener(external_protocol=elb.LoadBalancingProtocol.HTTP)  # Noncompliant
        lb.add_listener(external_protocol=elb.LoadBalancingProtocol.HTTPS)
        lb.add_listener(external_protocol=external_protocol)
        lb.add_listener()

        elb.LoadBalancer(listeners=[{"external_protocol": elb.LoadBalancingProtocol.HTTP}])  # Noncompliant
        elb.LoadBalancer(listeners=[{"external_protocol": elb.LoadBalancingProtocol.HTTPS}])

        listener = {"external_protocol": elb.LoadBalancingProtocol.HTTP}  # Noncompliant
        elb.LoadBalancer(listeners=[listener])

        random_dict = {"external_protocol": elb.LoadBalancingProtocol.HTTP}

        key = "external_protocol"
        value = elb.LoadBalancingProtocol.HTTP
        elb.LoadBalancer(listeners=[{key: value}])  # Noncompliant

class CfnLoadBalancerStack(Stack):
    def __init__(self, app: App, id: str) -> None:
        elb.CfnLoadBalancer.ListenersProperty(protocol="tcp")  # Noncompliant
        elb.CfnLoadBalancer.ListenersProperty(protocol="ssl")
        elb.CfnLoadBalancer.ListenersProperty(protocol=protocol)
        elb.CfnLoadBalancer.ListenersProperty()
