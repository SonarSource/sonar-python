from aws_cdk import aws_iam as iam

iam.PolicyStatement.from_json({
        'Action': 'iam:UpdateLoginProfile',
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Permissions are granted on all resources.}}
        'Effect': 'Allow',
# Noncompliant@+1 {{This policy is vulnerable to the "Update Login Profile " privilege escalation vector. Remove permissions or restrict the set of resources they apply to.}}
        'Resource': '*'
    #   ^^^^^^^^^^^^^^^
    })

iam.PolicyStatement.from_json({
        'Action': 'iam:UpdateLoginProfile',
        'Effect': 'Allow',
        'Resource': ['*']  # Noncompliant
    })

iam.PolicyStatement.from_json({
        'Action': ['iam:UpdateLoginProfile'],
        'Effect': 'Allow',
        'Resource': '*'  # Noncompliant
    })

iam.PolicyStatement.from_json({
        'Action': ['iam:UpdateLoginProfile'],
        'Resource': '*'  # Noncompliant
    },
)

iam.PolicyStatement.from_json({
        'Action': 'iam:UpdateLoginProfile',
        'Effect': 'Deny',
        'Resource': '*'
    })

iam.PolicyStatement.from_json({
        'Action': 'iam:FooBar',
        'Effect': 'Allow',
        'Resource': '*'
    })

iam.PolicyStatement.from_json({
        'Action': ['iam:UpdateLoginProfile'],
        'Effect': 'Allow',
        'Resource': '*',
        'Principal': ''
    })

iam.PolicyStatement.from_json({
        'Action': ['iam:UpdateLoginProfile'],
        'Effect': 'Allow',
        'Resource': '*',
        'Condition': ''
    })

