import boto3
import aiobotocore.session

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_data(Namespace='AWS/MyCustomService')  # Noncompliant {{Do not use AWS reserved namespace that begins with 'AWS/' for custom metrics.}}
#                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cloudwatch.put_metric_data(Namespace='OK/MyCustomService')
cloudwatch.put_metric_data()

def lambda_handler(event, context):
    cloudwatch.put_metric_data(Namespace='AWS/MyCustomService')  # Noncompliant

async def publish_metrics():
    session = aiobotocore.session.get_session()
    client = session.create_client('cloudwatch')
    client.put_metric_data(Namespace='AWS/Lambda/Custom') # Noncompliant

    async with session.create_client('cloudwatch') as with_client:
        await with_client.put_metric_data(Namespace='AWS/Lambda/Custom') # FN