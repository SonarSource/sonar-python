import os

def os_environ_lambda_handler(event, context):

    os.environ['_HANDLER'] = "lambda_function.lambda_handler"  # Noncompliant
    os.environ['_X_AMZN_TRACE_ID'] = "Root=1-67891233-abcdef012345678912345678;Sampled=1"  # Noncompliant
    os.environ['AWS_DEFAULT_REGION'] = "us-east-1"  # Noncompliant
    os.environ['AWS_REGION'] = "us-west-2"  # Noncompliant
    os.environ['AWS_EXECUTION_ENV'] = "AWS_Lambda_python3.8"  # Noncompliant
    os.environ['AWS_LAMBDA_FUNCTION_NAME'] = "my_lambda_function"  # Noncompliant
    os.environ['AWS_LAMBDA_FUNCTION_MEMORY_SIZE'] = "128"  # Noncompliant
    os.environ['AWS_LAMBDA_FUNCTION_VERSION'] = "$LATEST"  # Noncompliant
    os.environ['AWS_LAMBDA_INITIALIZATION_TYPE'] = "on-demand"  # Noncompliant
    os.environ['AWS_LAMBDA_LOG_GROUP_NAME'] = "/aws/lambda/my_lambda_function"  # Noncompliant
    os.environ['AWS_LAMBDA_LOG_STREAM_NAME'] = "2023/10/01/[$LATEST]abcdef1234567890"  # Noncompliant
    os.environ['AWS_ACCESS_KEY'] = "example_access_key"  # Noncompliant
    os.environ['AWS_ACCESS_KEY_ID'] = "AKIAEXAMPLE"  # Noncompliant
    os.environ['AWS_SECRET_ACCESS_KEY'] = "example_secret_key"  # Noncompliant
    os.environ['AWS_SESSION_TOKEN'] = "example_session_token"  # Noncompliant
    os.environ['AWS_LAMBDA_RUNTIME_API'] = "127.0.0.1:9001"  # Noncompliant
    os.environ['LAMBDA_TASK_ROOT'] = "/var/task"  # Noncompliant
    os.environ['LAMBDA_RUNTIME_DIR'] = "/var/runtime"  # Noncompliant

    os.environ['PATH'] = "/path"
    os.another_array['AWS_REGION'] = "us-east-1"


def multi_assignment_lambda_handler(event, context):
    smth, os.environ['AWS_REGION'] = 1, "us-west-2"  # Noncompliant
    return {"statusCode": 200}

def without_string_literal_lambda_handler(event, context):
    from a_module import importred_region_str
    region_str = "AWS_REGION"
    int_var = 42
    os.environ[region_str] = "us-west-2"  # Noncompliant
    os.environ[importred_region_str] = "us-west-2"
    os.environ[42] = "smth"
    os.environ[int_var] = "smth"

def environ_assignment_lambda_handler(event, context):
    from os import environ
    environ['AWS_REGION'] = 1, "us-west-2"  # Noncompliant
    smth, environ['AWS_REGION'] = 1, "us-west-2"  # Noncompliant
    return {"statusCode": 200}

def not_a_lambda_handler():
    # Outside of a Lambda handler
    os.environ['AWS_REGION'] = "us-west-2"  # Compliant

os.environ['AWS_REGION'] = "us-west-2"  # Compliant