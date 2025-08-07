import os
import tempfile

# Lambda handler - non-compliant cases
def with_open_lambda_handler(event, context): 
#   ^^^^^^^^^^^^^^^^^^^^^^^^> {{The temporary folder is used in this Lambda function.}}
    file_path = '/tmp/temp_data.txt'
    with open(file_path, 'w') as f:  # Noncompliant {{Clean up this temporary file before the Lambda function completes.}}
    #    ^^^^
        f.write("Something")

def open_lambda_handler(event, context):  
    file_path = '/tmp/another_file.txt'
    f = open(file_path, 'w')  # Noncompliant
    f.write("Data")
    f.close()

def os_remove_other_path_lambda_handler(event, context):
    file_path = '/tmp/temp_data.txt'
    with open(file_path, 'w') as f: # Noncompliant
        f.write("Something")
    os.remove("somefile.txt")

def os_unlink_no_path_lambda_handler(event, context):
    file_path = '/tmp/temp_data.txt'
    with open(file_path, 'w') as f: # Noncompliant
        f.write("Something")
    os.unlink()



# Lambda handler - compliant cases
def os_remove_lambda_handler(event, context):
    file_path = '/tmp/temp_data.txt'
    with open(file_path, 'w') as f:
        f.write("Something")
    os.remove(file_path)


def cleanup_in_other_function_lambda_handler(event, context):
    clean_up_path = '/tmp/temp_data.txt'
    with open(clean_up_path, 'w') as f: 
        f.write("Something")
    cleanup(clean_up_path) 

def cleanup(file_path):
    os.unlink(file_path)

def incorrect_cleanup_lambda_handler(event, context):
    clean_up_path = '/tmp/temp_data.txt'
    with open(clean_up_path, 'w') as f:  # FN as we stop the check once the file path is passed to a function
        f.write("Something")
    incorrect_cleanup(clean_up_path) 

def incorrect_cleanup(file_path):
    return file_path

def os_unlink_lambda_handler(event, context):
    # Compliant: /tmp file with proper cleanup using os.unlink
    file_path = '/tmp/temp_data.txt'
    f = open(file_path, 'w')
    f.write("Data")
    f.close()
    os.unlink(file_path)


def tempfile_lambda_handler(event, context):
    with tempfile.TemporaryFile() as temp_file:
        temp_file.write(b"Some data")

# Lambda handler with non-/tmp paths - should not trigger
def not_tmp_lambda_handler(event, context):
    # Compliant: not writing to /tmp
    file_path = '/var/log/app.log'
    with open(file_path, 'w') as f:
        f.write("Log data")

# Non-lambda functions - should not trigger issues
def regular_function():
    file_path = '/tmp/temp_data.txt'
    with open(file_path, 'w') as f:
        f.write("Something")


# =============== Coverage =================


def no_path_lambda_handler(event, context):
    with open() as f:
        f.write("Log data")

def not_string_path_lambda_handler(event, context):
    with open(2, 'w') as f:
        f.write("Log data")

def not_string_var_path_lambda_handler(event, context):
    file_path = 2
    with open(file_path, 'w') as f:
        f.write("Log data")

def different_string_lambda_handler(event, context):
    with open("/tmp/file.txt", 'w') as f: 
        f.write("Something")
    os.remove("/tmp/other_file.txt") 

def same_string_lambda_handler(event, context):
    with open("/tmp/file.txt", 'w') as f: 
        f.write("Something")
    os.remove("/tmp/file.txt") 
