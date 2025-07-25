import aws_cdk.aws_iam as iam
import aws_cdk.aws_s3 as s3

no_config = s3.Bucket(self, "bucket")  # Noncompliant {{No bucket policy enforces HTTPS-only access to this bucket. Make sure it is safe here.}}
#           ^^^^^^^^^

ssl_false = s3.Bucket(self, "bucket", enforce_ssl=False)  # Noncompliant {{Make sure authorizing HTTP requests is safe here.}}
#           ^^^^^^^^^


with_config = s3.Bucket(self, "bucket")

result = with_config.add_to_resource_policy(iam.PolicyStatement(  # Noncompliant {{Make sure authorizing HTTP requests is safe here.}}
        #                                   ^^^^^^^^^^^^^^^^^^^
        effect=iam.Effect.DENY,
        resources=[bucket.bucket_arn],
        actions=["s3:SomeAction"],
        principals=[roles],
        conditions=[{"Bool": {"aws:SecureTransport": False}}],
    )
)

empty_policy_call = s3.Bucket(self, "bucket") # Noncompliant
result = empty_policy_call.add_to_resource_policy()

no_policy_added = s3.Bucket(self, "bucket") # Noncompliant
result = no_policy_added.foo( 
    iam.PolicyStatement(  
        effect=iam.Effect.DENY,
        resources=["*"],
        actions=["s3:*"],
        principals=["*"],
        conditions=["SecureTransport:False"],
    )
)

not_policy_statement = s3.Bucket(self, "bucket") # Noncompliant
result = not_policy_statement.add_to_resource_policy( 
    iam.Foo(  
        effect=iam.Effect.DENY,
        resources=["*"],
        actions=["s3:*"],
        principals=["*"],
        conditions=["SecureTransport:False"],
    )
)
from module import foo

ssl_true = s3.Bucket(self, "bucket", enforce_ssl=True)  # Compliant
ssl_unknown = s3.Bucket(self, "bucket", enforce_ssl=foo())  # Compliant

correct_policy = s3.Bucket(self, "bucket")
result = correct_policy.add_to_resource_policy(
    iam.PolicyStatement(  # Compliant
        effect=iam.Effect.DENY,
        resources=["*"],
        actions=["s3:*"],
        principals=["*"],
        conditions=["SecureTransport:False"],
    )
)

compliant_policy = s3.Bucket(self, "bucket")
result = compliant_policy.add_to_resource_policy(
    iam.PolicyStatement(  # Compliant
        effect=iam.Effect.DENY,
        resources=["foo", "*"],
        actions=["s3:*", "foo:foo"],
        principals=[
            "role",
            "other_role",
            "*",
        ],
        conditions=["condition_a", "SecureTransport:False"],
    )
)

# ==================== COVERAGE ====================


a = s3.something(self, "bucket")
result = a.add_to_resource_policy(
    iam.PolicyStatement(  # Compliant it is not on a Bucket
        effect=iam.Effect.DENY,
        resources=[bucket.bucket_arn],
        actions=["s3:SomeAction"],
        principals=[roles],
        conditions=[{"Bool": {"aws:SecureTransport": False}}],
    )
)

b, c = s3.something(self, "bucket"), "test"  # We do not track multiple assignements
b = c = s3.something(self, "bucket")

d = {}

# FP we cannot resolve d["foo"] symbol
d["foo"] = s3.Bucket(self, "bucket")  # Noncompliant

d["foo"].add_to_resource_policy(
    iam.PolicyStatement(
        effect=iam.Effect.DENY,
        resources=["*"],
        actions=["s3:*"],
        principals=["*"],
        conditions=["SecureTransport:False"],
    )
)


# FP we do not track the policy variable
policy_in_var = s3.Bucket(self, "bucket") # Noncompliant
policy = iam.PolicyStatement(
    effect=iam.Effect.DENY,
    resources=["*"],
    actions=["s3:*"],
    principals=["*"],
    conditions=["SecureTransport:False"],
)
result = policy_in_var.add_to_resource_policy(policy)

partial_policy_actions = s3.Bucket(self, "bucket")
result = partial_policy_actions.add_to_resource_policy(
    iam.PolicyStatement(  # Noncompliant
        effect=iam.Effect.DENY,
        resources=["*"],
        actions=["s3:SomeAction"],
        principals=["*"],
        conditions=["SecureTransport:False"],
    )
)

partial_policy_conditions = s3.Bucket(self, "bucket")
result = partial_policy_conditions.add_to_resource_policy(
    iam.PolicyStatement(  # Noncompliant
        effect=iam.Effect.DENY,
        resources=["*"],
        actions=["s3:*"],
        principals=["*"],
        conditions=["Test"],
    )
)

partial_policy_effect = s3.Bucket(self, "bucket")
result = partial_policy_effect.add_to_resource_policy(
    iam.PolicyStatement(  # Noncompliant
        effect=iam.Effect.ALLOW,
        resources=["*"],
        actions=["s3:*"],
        principals=["*"],
        conditions=["SecureTransport:False"],
    )
)

partial_policy_roles = s3.Bucket(self, "bucket")
result = partial_policy_roles.add_to_resource_policy(
    iam.PolicyStatement(  # Noncompliant
        effect=iam.Effect.DENY,
        resources=["*"],
        actions=["s3:*"],
        principals=[roles],
        conditions=["SecureTransport:False"],
    )
)

partial_policy_resources = s3.Bucket(self, "bucket")
result = partial_policy_resources.add_to_resource_policy(
    iam.PolicyStatement(  # Noncompliant
        effect=iam.Effect.DENY,
        resources=["foo"],
        actions=["s3:*"],
        principals=["*"],
        conditions=["SecureTransport:False"],
    )
)

