from aws_cdk import aws_iam as iam

policy_document = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowActionsList",
            "Effect": "Allow",
        #   ^^^^^^^^^^^^^^^^^>  {{Related effect}}
            "Action": "*",  # Noncompliant {{Make sure granting all privileges is safe here.}}
        #             ^^^
            "Resource": ["arn:aws:iam:::user/*"]
        },
        {
            "Sid": "AllowActionsStar",
            "Effect": "Allow",
            "Action": ["*"],  # Noncompliant
            "Resource": ["arn:aws:iam:::user/*"]
        },
        {
            "Sid": "AllowSomeActions",
            "Effect": "Allow",
            "Action": ["iam:GetAccountSummary"],  # Compliant
            "Resource": ["arn:aws:iam:::user/*"]
        }
    ]
}

policy_document_from_json = iam.PolicyDocument.from_json(policy_document)
