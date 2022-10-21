from aws_cdk import aws_iam as iam

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
  # ^^^^^^^^^^^^^^^^^^^^^^^> {{Related effect}}
    actions=["*"],  # Noncompliant {{Make sure granting all privileges is safe here.}}
  #          ^^^
)


iam.PolicyStatement(
    actions=["*"],  # Noncompliant {{Make sure granting all privileges is safe here.}}
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:GetAccountSummary"],
)

iam.PolicyStatement(
    effect=iam.Effect.DENY,
    actions=["*"],
)


iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["service:permission", "*"],  # Noncompliant {{Make sure granting all privileges is safe here.}}
  #                                ^^^
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
)
