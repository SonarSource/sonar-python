import aws_cdk.aws_elasticloadbalancingv2 as elbv2


class ApplicationListenerStack(Stack):
    def __init__(self, app: App, id: str) -> None:
        lb = elbv2.ApplicationLoadBalancer()
        # Noncompliant@+1 {{Make sure that using network protocols without an SSL/TLS underlay is safe here.}}
        lb.add_listener(port=80)
      # ^^^^^^^^^^^^^^^
        lb.add_listener(port=8080)  # Noncompliant

        # Noncompliant@+1 {{Make sure that using network protocols without an SSL/TLS underlay is safe here.}}
        lb.add_listener(protocol=elbv2.ApplicationProtocol.HTTP)
                      # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        lb.add_listener(
            protocol=elbv2.ApplicationProtocol.HTTP,  # Noncompliant
          # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            port=8080
        )

        lb = elbv2.ApplicationLoadBalancer()
        lb.add_listener(port=unknown)
        lb.add_listener(
            protocol=elbv2.ApplicationProtocol.HTTPS,
            port=443,
        )
        lb.add_listener(
            protocol=elbv2.ApplicationProtocol.HTTPS,
            port=8080,
        )
        lb.add_listener(port=8443)

        elbv2.ApplicationListener(port=8080)  # Noncompliant
        elbv2.ApplicationListener(protocol=elbv2.ApplicationProtocol.HTTP)  # Noncompliant
        elbv2.ApplicationListener(protocol=elbv2.ApplicationProtocol.HTTP, port=8080)  # Noncompliant

        elbv2.ApplicationListener(port=8443)
        elbv2.ApplicationListener(protocol=elbv2.ApplicationProtocol.HTTPS)
        elbv2.ApplicationListener(protocol=elbv2.ApplicationProtocol.HTTPS, port=8080)


class NetworkListenerStack(Stack):
    def __init__(self, app: App, id: str) -> None:
        elbv2.NetworkListener(protocol=elbv2.Protocol.TCP)  # Noncompliant
        elbv2.NetworkListener(protocol=elbv2.Protocol.TCP, certificates=[certificate])  # Noncompliant
        elbv2.NetworkListener(certificates=[])  # Noncompliant
        elbv2.NetworkListener()  # Noncompliant

        elbv2.NetworkListener(protocol=elbv2.Protocol.TLS)
        elbv2.NetworkListener(protocol=elbv2.Protocol.TLS, certificates=[])
        elbv2.NetworkListener(certificates=[certificate])
        elbv2.NetworkListener(certificates=certificates)

        lb = elbv2.NetworkLoadBalancer()
        lb.add_listener(protocol=elbv2.Protocol.TCP)  # Noncompliant
        lb.add_listener(protocol=elbv2.Protocol.TLS)


class CfnListenerStack(Stack):
    def __init__(self, app: App, id: str) -> None:
        elbv2.CfnListener(protocol="HTTP")  # Noncompliant
        elbv2.CfnListener(protocol="TCP")  # Noncompliant
        elbv2.CfnListener(protocol="UDP")  # Noncompliant
        elbv2.CfnListener(protocol="TCP_UDP")  # Noncompliant

        elbv2.CfnListener(protocol=protocol)
        elbv2.CfnListener(protocol="HTTPS")

