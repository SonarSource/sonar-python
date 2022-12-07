from aws_cdk import aws_iam as iam

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreatePolicyVersion"],
    resources=["*"]
)
