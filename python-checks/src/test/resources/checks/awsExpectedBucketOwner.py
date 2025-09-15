import aiobotocore.session
import boto3

# Create S3 clients
s3_client = boto3.client("s3")
session = aiobotocore.session.get_session()
aclient = session.create_client("s3")


def regular_function():
    bucket_name = "my-production-bucket"
    expected_owner = "123456789012"

    s3_client.get_object(  # Noncompliant {{Add the 'ExpectedBucketOwner' parameter to verify S3 bucket ownership.}}
#   ^^^^^^^^^^^^^^^^^^^^
        Bucket=bucket_name,
        Key="data.json",
    )

    s3_client.put_object(  # Noncompliant
        Bucket=bucket_name, Key="data.json", Body=b"some data"
    )

    s3_client.delete_object(  # Noncompliant
        Bucket=bucket_name, Key="data.json"
    )

    s3_client.head_object(  # Noncompliant
        Bucket=bucket_name, Key="data.json"
    )

    s3_client.list_objects(  # Noncompliant
        Bucket=bucket_name
    )

    s3_client.get_object_acl(  # Noncompliant
        Bucket=bucket_name, Key="data.json"
    )

    s3_client.put_object_acl(  # Noncompliant
        Bucket=bucket_name, Key="data.json", ACL="private"
    )

    s3_client.get_bucket_policy(  # Noncompliant
        Bucket=bucket_name
    )

    # Compliant - S3 operations with ExpectedBucketOwner
    s3_client.get_object(
        Bucket=bucket_name, Key="data.json", ExpectedBucketOwner=expected_owner
    )

    s3_client.put_object(
        Bucket=bucket_name,
        Key="data.json",
        Body=b"some data",
        ExpectedBucketOwner=expected_owner,
    )

    s3_client.delete_object(
        Bucket=bucket_name, Key="data.json", ExpectedBucketOwner=expected_owner
    )

    s3_client.head_object(
        Bucket=bucket_name, Key="data.json", ExpectedBucketOwner=expected_owner
    )

    s3_client.list_objects(Bucket=bucket_name, ExpectedBucketOwner=expected_owner)

    s3_client.list_objects_v2(Bucket=bucket_name, ExpectedBucketOwner=expected_owner)

    s3_client.get_object_acl(
        Bucket=bucket_name, Key="data.json", ExpectedBucketOwner=None
    )

    # Compliant - No bucket owner to verify against when creating a bucket
    s3_client.create_bucket(Bucket="new-bucket")

def compliant_when_wrapped(*args, **kwargs):
    s3_client.get_object(*args)
    s3_client.get_object(**kwargs)
    s3_client.get_object(*args, **kwargs)

def upload_download_file():
    bucket_name = "my-production-bucket"
    s3_client.upload_file("name", bucket_name, "1") # Noncompliant {{Add the 'ExpectedBucketOwner' to the 'ExtraArgs' parameter to verify S3 bucket ownership.}}
    s3_client.upload_fileobj("name", bucket_name, "1") # Noncompliant
    s3_client.download_file("name", bucket_name, "1") # Noncompliant
    s3_client.download_fileobj("name", bucket_name, "1") # Noncompliant

    # ExpectedBucketOwner is not an argument download_fileobj
    s3_client.download_fileobj("name", bucket_name, "1", ExpectedBucketOwner="") # Noncompliant 

    s3_client.upload_file("name", bucket_name, "1", ExtraArgs={}) # Compliant
    s3_client.upload_fileobj("name", bucket_name, "1", ExtraArgs=unknown()) # Compliant
    dict_config = {}
    s3_client.download_file("name", bucket_name, "1", ExtraArgs=dict_config) # Compliant
    s3_client.download_fileobj("name", bucket_name, "1", ExtraArgs={"ExpectedBucketOwner":None}) # Compliant

    # We should still raise on the other FQN
    s3_client.get_object(Bucket=bucket_name, ExtraArgs={}) # Noncompliant 

async def async_function():
    bucket_name = "my-production-bucket"
    expected_owner = "123456789012"

    await aclient.get_object(  # Noncompliant {{Add the 'ExpectedBucketOwner' parameter to verify S3 bucket ownership.}}
        # ^^^^^^^^^^^^^^^^^^
        Bucket=bucket_name,
        Key="data.json",
    )

    await aclient.put_object(  # Noncompliant
        Bucket=bucket_name, Key="data.json", Body=b"some data"
    )

    await aclient.put_object_acl(  # Noncompliant
        Bucket=bucket_name, Key="data.json", ACL="private"
    )

    # Compliant - aiobotocore S3 operations with ExpectedBucketOwner
    await aclient.get_object(
        Bucket=bucket_name, Key="data.json", ExpectedBucketOwner=expected_owner
    )

    await aclient.put_object(
        Bucket=bucket_name,
        Key="data.json",
        Body=b"some data",
        ExpectedBucketOwner=None,
    )


def some_lambda_handler(event, context):
    s3_client.get_object(Bucket="bucket", Key="key") # Noncompliant

    s3_client.put_object(Bucket="bucket", Key="key", Body=b"data") # Noncompliant


# Non-S3 operations in lambda handler - should not be flagged
def another_lambda_handler(event, context):
    # These should not trigger issues since they're not S3 operations
    ec2_client = boto3.client("ec2")
    ec2_client.describe_instances()

