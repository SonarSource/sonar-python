from aws_cdk import aws_iam as iam

# PolicyStatement constructor with ALLOW effect and sensitive principals

iam.PolicyStatement(
  sid="AllowAnyPrincipal",
  effect=iam.Effect.ALLOW, # {{Access is set to "ALLOW" here.}}
        #^^^^^^^^^^^^^^^^>
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.StarPrincipal()] # Noncompliant {{Make sure granting public access is safe here.}}
             #^^^^^^^^^^^^^^^^^^^
)

iam.PolicyStatement( # {{Access is set to "ALLOW" here as default}}
  sid="AllowAnyPrincipal",
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.StarPrincipal()] # Noncompliant {{Make sure granting public access is safe here.}}
)

iam.PolicyStatement(
  sid="AllowAnyPrincipal",
  effect=iam.Effect.ALLOW,
        #^^^^^^^^^^^^^^^^>
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.AnyPrincipal()] # Noncompliant {{Make sure granting public access is safe here.}}
             #^^^^^^^^^^^^^^^^^^
)

iam.PolicyStatement(
  sid="AllowAnyPrincipal",
  effect=iam.Effect.ALLOW,
        #^^^^^^^^^^^^^^^^>
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.ArnPrincipal("*")] # Noncompliant
             #^^^^^^^^^^^^^^^^^^^^^
)

iam.PolicyStatement(
  sid="AllowAnyPrincipal",
  effect=iam.Effect.ALLOW,
        #^^^^^^^^^^^^^^^^>
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.AccountRootPrincipal(), iam.ArnPrincipal("*")] # Noncompliant
             #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
)

# PolicyStatement constructor with ALLOW effect and non-sensitive principals

iam.PolicyStatement(
  sid="AllowAccountRootPrincipal",
  effect=iam.Effect.ALLOW,
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.AccountRootPrincipal()] # Compliant
)

iam.PolicyStatement(
  sid="AllowAnyPrincipal",
  effect=iam.Effect.ALLOW,
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.ArnPrincipal("arn:aws:iam::123456789012:user/user-name")] # Compliant
)

# PolicyStatement constructor with DENY effect and sensitive principals

iam.PolicyStatement(
  sid="DenyAnyPrincipal",
  effect=iam.Effect.DENY,
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.StarPrincipal()] # Compliant
)
iam.PolicyStatement(
  sid="DenyAnyPrincipal",
  effect=iam.Effect.DENY,
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.AnyPrincipal()] # Compliant
)
iam.PolicyStatement(
  sid="DenyAnyPrincipal",
  effect=iam.Effect.DENY,
  actions=["s3:*"],
  resources=[bucket.arn_for_objects("*")],
  principals=[iam.ArnPrincipal("*")] # Compliant
)

# PolicyStatement from_json

iam.PolicyStatement.from_json({
    "Sid": "AllowAnyPrincipal",
    "Effect": "Allow",
             #^^^^^^^>
    "Action": ["s3:*"],
    "Resource": bucket.arn_for_objects("*"),
    "Principal": "*" # Noncompliant
   #^^^^^^^^^^^^^^^^
})

iam.PolicyStatement.from_json({
    "Sid": "AllowAnyPrincipal",
    "Effect": "Allow",
             #^^^^^^^>
    "Action": ["s3:*"],
    "Resource": bucket.arn_for_objects("*"),
    "Principal": { "AWS" : "*"} # Noncompliant
                  #^^^^^^^^^^^
})

iam.PolicyStatement.from_json({
    "Sid": "AllowAnyPrincipal",
    "Action": ["s3:*"],
    "Resource": bucket.arn_for_objects("*"),
    "Principal": { "AWS" : "*"} # Noncompliant
                  #^^^^^^^^^^^
})

iam.PolicyStatement.from_json({
    "Sid": "AllowAnyPrincipal",
    "Effect": "Allow",
             #^^^^^^^>
    "Action": ["s3:*"],
    "Resource": bucket.arn_for_objects("*"),
    "Principal": { "AWS" : [ "*", "9999999" ]} # Noncompliant
                            #^^^^^^^^^^^^^^
})

# PolicyStatement from_json, non-sensitive effect

iam.PolicyStatement.from_json({
    "Sid": "AllowAnyPrincipal",
    "Effect": "Deny",
    "Action": ["s3:*"],
    "Resource": bucket.arn_for_objects("*"),
    "Principal": { "AWS" : "*"} # Compliant
})

# PolicyStatement from_json, non-sensitive principals

iam.PolicyStatement.from_json({
    "Sid": "AllowAccountRootPrincipal",
    "Effect": "Allow",
    "Action": ["s3:*"],
    "Resource": bucket.arn_for_objects("*"),
    "Principal": { "AWS" : iam.AccountRootPrincipal().arn } # Compliant
})

# PolicyDocument from_json

iam.PolicyDocument.from_json({
 "Version": "2012-10-17",
 "Statement": [{
   "Sid": "AnyPrincipal",
   "Effect": "Allow",
            #^^^^^^^>
   "Action": ["kms:*"],
   "Resource": "*",
   "Principal": {
     "AWS": ["*", "999999999999"], # Noncompliant
            #^^^^^^^^^^^^^^^^^^^
   }
 },
 {
   "Sid": "AccountRootPrincipal",
   "Effect": "Allow",
   "Action": ["kms:*"],
   "Resource": "*",
   "Principal": { "AWS" : iam.AccountRootPrincipal().arn } # Compliant
 }]
})

# in a variable
policy_document = {
 "Version": "2012-10-17",
 "Statement": [{
   "Sid": "AnyPrincipal",
   "Effect": "Allow",
            #^^^^^^^>
   "Action": ["kms:*"],
   "Resource": "*",
   "Principal": {
     "AWS": ["*", "999999999999"], # Noncompliant
            #^^^^^^^^^^^^^^^^^^^
   }
 },
 {
   "Sid": "AccountRootPrincipal",
   "Effect": "Allow",
   "Action": ["kms:*"],
   "Resource": "*",
   "Principal": { "AWS" : iam.AccountRootPrincipal().arn } # Compliant
 },
 {
   "Sid": "AccountRootPrincipal",
   "Effect": "Deny",
   "Action": ["kms:*"],
   "Resource": "*",
   "Principal": { "AWS" : "*" } # Compliant
 },
 {
    "Sid": "AnyPrincipal",
    "Effect": "Allow",
             #^^^^^^^>
    "Action": ["kms:*"],
    "Resource": "*",
    "Principal": "*" # Noncompliant
   #^^^^^^^^^^^^^^^^
  },
  {
    "Sid": "AnyPrincipal",
    "Action": ["kms:*"],
    "Resource": "*",
    "Principal": "*" # Noncompliant
   #^^^^^^^^^^^^^^^^
  },
 ]
}

iam.PolicyDocument.from_json(policy_document)

# edge cases

iam.PolicyStatement(sid="AllowAnyPrincipal")
iam.PolicyStatement(
  effect="Whatever",
  principals=[iam.ArnPrincipal("*")]
)
iam.PolicyStatement(
  effect=iam.Effect.ALLOW,
  principals=[iam.ArnPrincipal()]
)

iam.PolicyStatement.from_json({})
iam.PolicyStatement.from_json()
iam.PolicyStatement.from_json({
    "Effect": "Allow",
})

iam.PolicyDocument.from_json({})
iam.PolicyDocument.from_json()
