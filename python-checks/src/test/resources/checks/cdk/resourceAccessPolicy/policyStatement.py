from aws_cdk import aws_iam as iam

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
#   ^^^^^^^^^^^^^^^^^^^^^^^> {{Related effect}}
    actions=["iam:CreatePolicyVersion"],
    resources=["*"]  # Noncompliant {{Make sure granting access to all resources is safe here.}}
#              ^^^
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreatePolicyVersion", "kms:*"],
    resources=["*"]  # Noncompliant
)

iam.PolicyStatement(
    actions=["iam:CreatePolicyVersion"],
    resources=["*"]  # Noncompliant
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreatePolicyVersion"],
    resources=[dummy_policy.managed_policy_arn]
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["kms:foo", "kms:bar"],
    resources=["*"]
)

iam.PolicyStatement(
    actions=["kms:*"],
    principals=[iam.AccountRootPrincipal()],
    resources=["*"]
)

iam.PolicyStatement(
    effect=iam.Effect.DENY,
    actions=["iam:CreatePolicyVersion"],
    resources=["*"]
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreatePolicyVersion", "kms:*"],
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    resources=["*"]
)
