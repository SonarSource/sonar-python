import aws_cdk.aws_ec2 as ec2

class Test:
    def test_connections_attribute(var):
        # checking for 'connections' attribute of a specific object
        obj = ec2.Instance()
        obj.connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant{{Change this IP range to a subset of trusted IP addresses.}}
#                                  ^^^^^^^^^^^^^^^^^^^
        obj.connections.allow_from_any_ipv4(ec2.Port.all_tcp()) # Noncompliant{{Change this method for `allow_from` and set `other` to a subset of trusted IP addresses.}}
#                       ^^^^^^^^^^^^^^^^^^^
        obj.connections.allow_from_any_ipv4(ec2.Port.tcp(20))

        # checking on direct call on newly created object
        ec2.Instance().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant

    def test_Connections_object(var):
        # checking directly on aws_cdk.aws_ec2.Connections objects
        connectionCompliantDefault = ec2.Connections(default_port=ec2.Port.tcp(443)) # compliant default port
        connectionSensitiveDefault = ec2.Connections(default_port=ec2.Port.all_tcp()) # sensitive default port

        connectionCompliantDefault.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant{{Change this IP range to a subset of trusted IP addresses.}}
        connectionSensitiveDefault.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant{{Change this IP range to a subset of trusted IP addresses.}}
        connectionCompliantDefault.allow_from_any_ipv4(ec2.Port.all_tcp()) # Noncompliant{{Change this method for `allow_from` and set `other` to a subset of trusted IP addresses.}}
        connectionSensitiveDefault.allow_from_any_ipv4(ec2.Port.all_tcp()) # Noncompliant{{Change this method for `allow_from` and set `other` to a subset of trusted IP addresses.}}
        connectionCompliantDefault.allow_default_port_from_any_ipv4()
        connectionSensitiveDefault.allow_default_port_from_any_ipv4() # Noncompliant{{Change this method for `allow_from` and set `other` to a subset of trusted IP addresses.}}
#                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        connectionCompliantDefault.allow_default_port_from(ec2.Peer.any_ipv4())
        connectionSensitiveDefault.allow_default_port_from(ec2.Peer.any_ipv4()) # Noncompliant{{Change this method for `allow_from` and set `other` to a subset of trusted IP addresses.}}
#                                  ^^^^^^^^^^^^^^^^^^^^^^^
        connectionCompliantDefault.allow_default_port_from(ec2.Peer.ipv4("192.168.1.1/24"))
        connectionSensitiveDefault.allow_default_port_from(ec2.Peer.ipv4("192.168.1.1/24"))

        # checking on direct call on newly created object
        ec2.Connections(default_port=ec2.Port.all_tcp()).allow_default_port_from_any_ipv4() # Noncompliant{{Change this method for `allow_from` and set `other` to a subset of trusted IP addresses.}}
#                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ec2.Connections(default_port=ec2.Port.all_tcp()).allow_default_port_from(ec2.Peer.any_ipv4()) # Noncompliant

    def test_SecurityGroup(var):
        security_group = ec2.SecurityGroup()
        security_group.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant{{Change this IP range to a subset of trusted IP addresses.}}
#                                       ^^^^^^^^^^^^^^^^^^^
        security_group.add_ingress_rule(peer=ec2.Peer.any_ipv4(), connection=ec2.Port.all_tcp()) # Noncompliant
        security_group.add_ingress_rule(connection=ec2.Port.all_tcp(), peer=ec2.Peer.any_ipv4()) # Noncompliant

        security_group.add_ingress_rule
        security_group.add_ingress_rule()
        security_group.add_ingress_rule("random value")
        security_group.add_ingress_rule(ec2.Peer.any_ipv4())
        security_group.add_ingress_rule(ec2.Port.all_tcp())

    def test_bad_port_variant(var):
        conn = ec2.Instance().connections

        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_traffic()) # Noncompliant
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp(22)) # Noncompliant
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp(port=3389)) # Noncompliant
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp_range(start_port=20, end_port=30)) # Noncompliant
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port(protocol=ec2.Protocol.TCP, from_port=20, to_port=30)) # Noncompliant
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port(protocol=ec2.Protocol.ALL, from_port=20, to_port=30)) # Noncompliant

        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp)
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp(50))
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp(variable=3389))
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp(port="3389"))
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp_range(start_port=30, end_port=40))
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp_range(start_port="20", end_port="30"))
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp_range)
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp_range(end_port=40))
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp_range(start_port=30))
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.tcp_range())
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port(protocol=ec2.Protocol.TCP, from_port=30, to_port=40))
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port(protocol=ec2.Protocol.ALL, from_port=30, to_port=40))
        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port(protocol=ec2.Protocol.UDP, from_port=20, to_port=30))

    def test_bad_peer_variant(var):
        conn = ec2.Instance().connections

        conn.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        conn.allow_from(ec2.Peer.any_ipv6(), ec2.Port.all_tcp()) # Noncompliant
        conn.allow_from(ec2.Peer.ipv4("0.0.0.0/0"), ec2.Port.all_tcp()) # Noncompliant
        conn.allow_from(ec2.Peer.ipv6("::/0"), ec2.Port.all_tcp()) # Noncompliant
        conn.allow_from(ec2.Peer.ipv4(cidr_ip="0.0.0.0/0"), ec2.Port.all_tcp()) # Noncompliant
        conn.allow_from(ec2.Peer.ipv6(cidr_ip="::/0"), ec2.Port.all_tcp()) # Noncompliant

        conn.allow_from(ec2.Peer.ipv4(""), ec2.Port.all_tcp())
        conn.allow_from(ec2.Peer.ipv6(""), ec2.Port.all_tcp())
        conn.allow_from(ec2.Peer.ipv4(), ec2.Port.all_tcp())
        conn.allow_from(ec2.Peer.ipv6(), ec2.Port.all_tcp())
        conn.allow_from(ec2.Peer.ipv4, ec2.Port.all_tcp())
        conn.allow_from(ec2.Peer.ipv6, ec2.Port.all_tcp())
        conn.allow_from(ec2.Peer.ipv4("192.168.1.1/24"), ec2.Port.all_tcp())
        conn.allow_from(ec2.Peer.ipv6("1.2.3.4.5.6/24"), ec2.Port.all_tcp())
        conn.allow_from(ec2.Peer.ipv4(other="0.0.0.0/0"), ec2.Port.all_tcp())
        conn.allow_from(ec2.Peer.ipv6(other="::/0"), ec2.Port.all_tcp())

    def test_all_supported_object_with_connections_attributes(var):
        aws_cdk.aws_docdb.DatabaseCluster().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_lambda_python_alpha.PythonFunction().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_batch_alpha.ComputeEnvironment().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_efs.FileSystem().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_lambda_go_alpha.GoFunction().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_ecs.ExternalService().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_ecs.FargateService().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_ecs.Cluster().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_ecs.Ec2Service().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_elasticsearch.Domain().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_neptune_alpha.DatabaseCluster().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_eks.FargateCluster().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_eks.Cluster().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_codebuild.PipelineProject().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_codebuild.Project().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_rds.DatabaseInstance().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_rds.DatabaseInstanceReadReplica().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_rds.DatabaseCluster().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_rds.ServerlessClusterFromSnapshot().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_rds.DatabaseProxy().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_rds.DatabaseInstanceFromSnapshot().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_rds.ServerlessCluster().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_rds.DatabaseClusterFromSnapshot().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_lambda_nodejs.NodejsFunction().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_fsx.LustreFileSystem().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_ec2.BastionHostLinux().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_ec2.ClientVpnEndpoint().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_ec2.Instance().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_ec2.LaunchTemplate().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_ec2.SecurityGroup().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_kinesisfirehose_alpha.DeliveryStream().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_stepfunctions_tasks.SageMakerCreateTrainingJob().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_stepfunctions_tasks.SageMakerCreateModel().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_stepfunctions_tasks.EcsRunTask().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_redshift_alpha.Cluster().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_opensearchservice.Domain().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_secretsmanager.HostedRotation().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_msk_alpha.Cluster().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.triggers.TriggerFunction().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_autoscaling.AutoScalingGroup().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_synthetics_alpha.Canary().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_cloudfront.experimental.EdgeFunction().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_lambda.Function().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_lambda.DockerImageFunction().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_lambda.SingletonFunction().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_lambda.Alias().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
        aws_cdk.aws_lambda.Version().connections.allow_from(ec2.Peer.any_ipv4(), ec2.Port.all_tcp()) # Noncompliant
