from aws_cdk import aws_iam as iam

iam.PolicyStatement.from_json({
    "Effect": "Allow",
 #  ^^^^^^^^^^^^^^^^^> {{Related effect}}
    "Action": ["*"],  # Noncompliant {{Make sure granting all privileges is safe here.}}
 #             ^^^
})

iam.PolicyStatement.from_json({
    "Effect": "Allow",
    "Action": "*",  # Noncompliant
 #  ^^^^^^^^^^^^^
})

iam.PolicyStatement.from_json({
    "Effect": "Allow",
    "Action": ["iam:GetAccountSummary"]
})

iam.PolicyStatement.from_json({
    "Effect": "Deny",
    "Action": ["*"],
})

iam.PolicyStatement.from_json({
    "Action": ["*"],
})

iam.PolicyStatement.from_json({
    "Effect": "Allow",
    "Action": ["iam:GetAccountSummary", "*"],  # Noncompliant
})
