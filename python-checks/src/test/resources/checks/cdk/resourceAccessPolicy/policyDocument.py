from aws_cdk import aws_iam as iam

iam.PolicyDocument.from_json({
    "Statement": [
        {
            "Effect": "ALLOW",
            "Action": ["iam:CreatePolicyVersion"],
            "Resource": ["*"]  # Noncompliant
        },
        {
            "Effect": "ALLOW",
            "Action": ["iam:CreatePolicyVersion"],
            "Resource": "*"  # Noncompliant
        },
        {
            "Effect": "ALLOW",
            "Action": ["bob:random_key"],
            "Resource": "*"
        },
        {
            "Effect": "ALLOW",
            "Action": ["bob:random_key", "iam:CreatePolicyVersion"],
            "Resource": "*" # Noncompliant
        },
        {
            "Effect": "ALLOW",
            "Action": ["iam:CreatePolicyVersion"],
            "Resource": [dummy_policy.managed_policy_arn]
        }
    ]
})
