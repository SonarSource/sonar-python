import boto3

client = boto3.client('s3')

def lambda_handler(event, context):
    while True:
        response = client.describe_instances()  # Noncompliant

    while context:
        response = client.describe_instances()  # OK

    response = client.describe_instances()  # OK


def not_a_lambda():
    while True:
        response = client.describe_instances()  # OK

    response = client.describe_instances()  # OK

while a:
    response = client.describe_instances()  # OK
