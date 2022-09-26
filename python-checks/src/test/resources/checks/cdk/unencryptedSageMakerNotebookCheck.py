from aws_cdk import (aws_sagemaker as sagemaker, RemovalPolicy)

class CfnSagemakerStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # A normal call to 'CfnNotebookInstance' look like this. For the sake of simplicity/readability, we will remove arguments not checked by the rule.
        sagemaker.CfnNotebookInstance(self, "Sensitive", instance_type="instanceType", role_arn="roleArn", kms_key_id=my_key.key_id)

        # variables
        my_key = kms.Key(self, "Key", removal_policy=RemovalPolicy.DESTROY)
        random_key_object = "a key ?"
        noneKey = None

        # Test case fail
        sagemaker.CfnNotebookInstance() # NonCompliant
        sagemaker.CfnNotebookInstance(kms_key_id=None) # NonCompliant
        sagemaker.CfnNotebookInstance(kms_key_id=noneKey) # NonCompliant

        # Test case success
        sagemaker.CfnNotebookInstance(kms_key_id=my_key.key_id)
        sagemaker.CfnNotebookInstance(kms_key_id=random_key_object)
        sagemaker.CfnNotebookInstance(kms_key_id=kms.Key(self, "Key", removal_policy=RemovalPolicy.DESTROY).key_id)
