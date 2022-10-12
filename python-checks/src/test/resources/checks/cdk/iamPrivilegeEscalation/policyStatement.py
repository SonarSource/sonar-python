from aws_cdk import aws_iam as iam

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreatePolicyVersion"],
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^> {{Permissions are granted on all resources.}}
    # Noncompliant@+1 {{This policy is vulnerable to the "Create Policy Version" privilege escalation vector. Remove permissions or restrict the set of resources they apply to.}}
    resources=["*"]
    #          ^^^
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreatePolicyVersion"],
    resources=["arn:a:b:c:d:role/*"]  # Noncompliant
)

iam.PolicyStatement(
    actions=["iam:CreatePolicyVersion"],
    resources=["*"]  # Noncompliant
)

iam.PolicyStatement(
    actions=["iam:CreatePolicyVersion"],
    resources=[attacker.user_arn]
)

iam.PolicyStatement(
    effect=iam.Effect.DENY,
    actions=["iam:CreatePolicyVersion"],
    resources=["*"]  # Ok
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["SuperSafeActionWithoutAnyVulnerabilities"],
    resources=["*"]  # Ok
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreatePolicyVersion"],
    resources=["*"],  # Ok, principals is defined
    principals="*"
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreatePolicyVersion"],
    resources=["*"],  # Ok, conditions is defined
    conditions=[]
)

# Attack vector names

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:SetDefaultPolicyVersion"],
    resources=["*"]  # Noncompliant {{This policy is vulnerable to the "Set Default Policy Version" privilege escalation vector. Remove permissions or restrict the set of resources they apply to.}}
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:SetDefaultPolicyVersion"],
    resources=["*"]  # Noncompliant {{This policy is vulnerable to the "Set Default Policy Version" privilege escalation vector. Remove permissions or restrict the set of resources they apply to.}}
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreateAccessKey"],
    resources=["*"]  # Noncompliant {{This policy is vulnerable to the "Create AccessKey" privilege escalation vector. Remove permissions or restrict the set of resources they apply to.}}
)

iam.PolicyStatement(
    effect=iam.Effect.ALLOW,
    actions=["iam:CreateLoginProfile"],
    resources=["*"]  # Noncompliant {{This policy is vulnerable to the "Create Login Profile" privilege escalation vector. Remove permissions or restrict the set of resources they apply to.}}
)
