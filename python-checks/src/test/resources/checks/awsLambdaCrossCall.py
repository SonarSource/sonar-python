import boto3

global_client = boto3.client('lambda')

def lambda_handler(event, context):
    global_client.invoke(InvocationType='RequestResponse') # Noncompliant
    global_client.invoke(InvocationType='some other value')
    global_client.invoke(FunctionName='target-lambda-function-name')

    local_client = boto3.client('lambda')
    local_client.invoke(InvocationType='RequestResponse') # Noncompliant
    local_client.invoke(InvocationType='some other value')
    local_client.invoke(FunctionName='target-lambda-function-name')

    other_client = "something else"
    other_client.invoke(InvocationType='RequestResponse')
    
    
def not_a_lambda():
    global_client.invoke(InvocationType='RequestResponse') # OK
    
    local_client = boto3.client('lambda')
    local_client.invoke(InvocationType='RequestResponse') # OK

global_client.invoke(InvocationType='RequestResponse') # OK

