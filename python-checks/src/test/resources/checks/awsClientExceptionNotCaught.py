import boto3
from botocore.exceptions import ClientError
import botocore.exceptions

s3 = boto3.client('s3')
ec2 = boto3.client('ec2')
lambda_client = boto3.client('lambda')
dynamodb = boto3.client('dynamodb')
rds = boto3.client('rds')
iam = boto3.client('iam')
cloudformation = boto3.client('cloudformation')
sns = boto3.client('sns')
sqs = boto3.client('sqs')

def noncompliant_examples_handler(event, ctx):
    # S3 operations
    s3.get_object(Bucket='bucket', Key='key')  # Noncompliant {{Wrap this AWS client call in a try-except block to handle "botocore.exceptions.ClientError".}}
#   ^^^^^^^^^^^^^
    s3.put_object(Bucket='bucket', Key='key', Body='data')  # Noncompliant
    s3.create_bucket(Bucket='bucket')  # Noncompliant
    s3.delete_object(Bucket='bucket', Key='key')  # Noncompliant
    s3.delete_bucket(Bucket='bucket')  # Noncompliant
    s3.list_objects_v2(Bucket='bucket')  # Noncompliant
    s3.copy_object(Bucket='dest', CopySource={'Bucket': 'src', 'Key': 'key'}, Key='key')  # Noncompliant
    s3.head_object(Bucket='bucket', Key='key')  # Noncompliant
    s3.get_bucket_location(Bucket='bucket')  # Noncompliant
    s3.put_bucket_policy(Bucket='bucket', Policy='policy')  # Noncompliant
    s3.get_bucket_policy(Bucket='bucket')  # Noncompliant
    s3.delete_bucket_policy(Bucket='bucket')  # Noncompliant
    
    # EC2 operations
    ec2.describe_instances()  # Noncompliant
    ec2.run_instances(ImageId='ami-12345', MinCount=1, MaxCount=1)  # Noncompliant
    ec2.terminate_instances(InstanceIds=['i-12345'])  # Noncompliant
    ec2.start_instances(InstanceIds=['i-12345'])  # Noncompliant
    ec2.stop_instances(InstanceIds=['i-12345'])  # Noncompliant
    ec2.create_security_group(GroupName='sg-name', Description='desc')  # Noncompliant
    ec2.delete_security_group(GroupId='sg-12345')  # Noncompliant
    ec2.describe_security_groups()  # Noncompliant
    ec2.create_vpc(CidrBlock='10.0.0.0/16')  # Noncompliant
    ec2.delete_vpc(VpcId='vpc-12345')  # Noncompliant
    ec2.describe_vpcs()  # Noncompliant
    
    # Lambda operations
    lambda_client.create_function(FunctionName='func', Runtime='python3.9', Role='role', Handler='handler', Code={'ZipFile': b'code'})  # Noncompliant
    lambda_client.update_function_code(FunctionName='func', ZipFile=b'code')  # Noncompliant
    lambda_client.update_function_configuration(FunctionName='func', Handler='new_handler')  # Noncompliant
    lambda_client.delete_function(FunctionName='func')  # Noncompliant
    lambda_client.invoke(FunctionName='func')  # Noncompliant
    lambda_client.get_function(FunctionName='func')  # Noncompliant
    lambda_client.list_functions()  # Noncompliant
    
    # DynamoDB operations
    dynamodb.get_item(TableName='table', Key={'id': {'S': 'value'}})  # Noncompliant
    dynamodb.put_item(TableName='table', Item={'id': {'S': 'value'}})  # Noncompliant
    dynamodb.delete_item(TableName='table', Key={'id': {'S': 'value'}})  # Noncompliant
    dynamodb.update_item(TableName='table', Key={'id': {'S': 'value'}}, UpdateExpression='SET attr = :val', ExpressionAttributeValues={':val': {'S': 'newvalue'}})  # Noncompliant
    dynamodb.query(TableName='table', KeyConditionExpression='id = :id', ExpressionAttributeValues={':id': {'S': 'value'}})  # Noncompliant
    dynamodb.scan(TableName='table')  # Noncompliant
    dynamodb.create_table(TableName='table', KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}], AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}], BillingMode='PAY_PER_REQUEST')  # Noncompliant
    dynamodb.delete_table(TableName='table')  # Noncompliant
    dynamodb.describe_table(TableName='table')  # Noncompliant
    
    # RDS operations
    rds.create_db_instance(DBInstanceIdentifier='db', DBInstanceClass='db.t3.micro', Engine='mysql')  # Noncompliant
    rds.delete_db_instance(DBInstanceIdentifier='db', SkipFinalSnapshot=True)  # Noncompliant
    rds.describe_db_instances()  # Noncompliant
    rds.modify_db_instance(DBInstanceIdentifier='db', AllocatedStorage=100)  # Noncompliant
    rds.reboot_db_instance(DBInstanceIdentifier='db')  # Noncompliant
    
    # IAM operations
    iam.create_user(UserName='user')  # Noncompliant
    iam.delete_user(UserName='user')  # Noncompliant
    iam.get_user(UserName='user')  # Noncompliant
    iam.create_role(RoleName='role', AssumeRolePolicyDocument='policy')  # Noncompliant
    iam.delete_role(RoleName='role')  # Noncompliant
    iam.get_role(RoleName='role')  # Noncompliant
    iam.attach_user_policy(UserName='user', PolicyArn='arn:aws:iam::policy')  # Noncompliant
    iam.detach_user_policy(UserName='user', PolicyArn='arn:aws:iam::policy')  # Noncompliant
    
    # CloudFormation operations
    cloudformation.create_stack(StackName='stack', TemplateBody='template')  # Noncompliant
    cloudformation.delete_stack(StackName='stack')  # Noncompliant
    cloudformation.describe_stacks()  # Noncompliant
    cloudformation.update_stack(StackName='stack', TemplateBody='template')  # Noncompliant
    
    # SNS operations
    sns.create_topic(Name='topic')  # Noncompliant
    sns.delete_topic(TopicArn='arn:aws:sns:topic')  # Noncompliant
    sns.publish(TopicArn='arn:aws:sns:topic', Message='message')  # Noncompliant
    sns.subscribe(TopicArn='arn:aws:sns:topic', Protocol='email', Endpoint='email@example.com')  # Noncompliant
    sns.unsubscribe(SubscriptionArn='arn:aws:sns:subscription')  # Noncompliant
    
    # SQS operations
    sqs.create_queue(QueueName='queue')  # Noncompliant
    sqs.delete_queue(QueueUrl='https://sqs.region.amazonaws.com/account/queue')  # Noncompliant
    sqs.send_message(QueueUrl='https://sqs.region.amazonaws.com/account/queue', MessageBody='message')  # Noncompliant
    sqs.receive_message(QueueUrl='https://sqs.region.amazonaws.com/account/queue')  # Noncompliant
    sqs.delete_message(QueueUrl='https://sqs.region.amazonaws.com/account/queue', ReceiptHandle='handle')  # Noncompliant
    
    try:
#   ^^^> {{This try does not catch the ClientError.}}
    
        s3.get_object(Bucket='bucket', Key='key')  # Noncompliant
    #   ^^^^^^^^^^^^^
    except ValueError:
        pass

    try:
        try:
    #   ^^^> {{This try does not catch the ClientError.}}
        
            s3.get_object(Bucket='bucket', Key='key')  # Noncompliant
        #   ^^^^^^^^^^^^^
        except ValueError:
            pass
    except SomeException:
        pass

def compliant_with_different_caught_exceptions_handler(event, ctx):
    try:
        s3.get_object(Bucket='bucket', Key='key')
    except ClientError:
        pass

    try:
        ec2.describe_instances()
    except Exception:
        pass

    try:
        s3.create_bucket(Bucket='bucket')
    except BaseException:
        pass

def compliant_delete_object_with_bare_except_handler(event, ctx):
    try:
        s3.delete_object(Bucket='bucket', Key='key')
    except:
        pass

def compliant_get_object_with_multiple_exceptions_handler(event, ctx):
    try:
        s3.get_object(Bucket='bucket', Key='key')
    except (ClientError, ValueError):
        pass

def compliant_nested_try_except_handler(event, ctx):
    try:
        try: 
            s3.get_object(Bucket='bucket', Key='key')
        except ValueError:
            pass
    except ClientError:
        pass

def other_function():
    s3.get_object(Bucket='bucket', Key='key') # FN, rule only raises on calls in a lambda handler

def compliant_called_from_another_function_handler(event, ctx):
    other_function()

