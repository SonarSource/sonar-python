from aws_cdk import aws_iam as iam

iam.PolicyDocument.from_json({
    'Statement': [
        {
            'Action': 'iam:UpdateLoginProfile',
        #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Permissions are granted on all resources.}}
            'Effect': 'Allow',
            # Noncompliant@+1 {{This policy is vulnerable to the "Update Login Profile " privilege escalation vector. Remove permissions or restrict the set of resources they apply to.}}
            'Resource': '*'
        #   ^^^^^^^^^^^^^^^
        },
        {
            'Action': 'iam:UpdateLoginProfile',
            'Effect': 'Allow',
            'Resource': ['*']  # Noncompliant
        },
        {
            'Action': ['iam:UpdateLoginProfile'],
            'Effect': 'Allow',
            'Resource': '*'  # Noncompliant
        },
        {
            'Action': ['iam:UpdateLoginProfile'],
            'Resource': '*'  # Noncompliant
        },
        {
            'Action': 'iam:UpdateLoginProfile',
            'Effect': 'Deny',
            'Resource': '*'
        },
        {
            'Action': 'iam:FooBar',
            'Effect': 'Allow',
            'Resource': '*'
        },
        {
            'Action': ['iam:UpdateLoginProfile'],
            'Effect': 'Allow',
            'Resource': '*',
            'Principal': ''
        },
        {
            'Action': ['iam:UpdateLoginProfile'],
            'Effect': 'Allow',
            'Resource': '*',
            'Condition': ''
        }
    ]
})
