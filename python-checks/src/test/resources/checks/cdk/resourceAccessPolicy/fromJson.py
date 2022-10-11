from aws_cdk import aws_iam as iam

iam.PolicyStatement.from_json({
    "Effect": "ALLOW",
#   ^^^^^^^^^^^^^^^^^> {{Related effect}}
    "Action": ["iam:CreatePolicyVersion"],
    "Resource": ["*"]  # Noncompliant {{Make sure granting access to all resources is safe here.}}
            #    ^^^
})

iam.PolicyStatement.from_json({
    "Effect": "ALLOW",
    "Action": ["iam:CreatePolicyVersion"],
    "Resource": "*"  # Noncompliant
})

iam.PolicyStatement.from_json({
    "Action": ["iam:CreatePolicyVersion"],
    "Resource": "*"  # Noncompliant
})

iam.PolicyStatement.from_json({
    "Effect": "ALLOW",
    "Action": ["iam:CreatePolicyVersion"],
    "Resource": [dummy_policy.managed_policy_arn]
})

iam.PolicyStatement.from_json({
    "Effect": "ALLOW",
    "Action": ["kms:foo", "kms:bar"],
    "Resource": "*"
})

iam.PolicyStatement.from_json({
    "Action": ["kms:*"],
    "Resource": "*"
})
