from aws_cdk.aws_ec2 import Volume
from aws_cdk import Stack
from constructs import Construct


class NonCompliantStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        Volume(self, encrypted=False)  # Noncompliant {{Make sure that using unencrypted volumes is safe here.}}
    #                ^^^^^^^^^^^^^^^

        Volume(self)  # Noncompliant {{Omitting "encrypted" disables volumes encryption. Make sure it is safe here.}}
    #   ^^^^^^

        encrypted = False
    #   ^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
        Volume(self, encrypted=encrypted)  # Noncompliant {{Make sure that using unencrypted volumes is safe here.}}
    #                ^^^^^^^^^^^^^^^^^^^

        # Noncompliant@+1
        Volume(self, "unencrypted-explicit", availability_zone="eu-west-1a", size=Size.gibibytes(1), encrypted=False)

        volume_args = {"encrypted": False}
        Volume(self, **volume_args)  # Noncompliant

class CompliantStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:

        Volume(self, encrypted=True)

        Volume(self, encrypted=unknown)

        encrypted = True
        Volume(self, encrypted=encrypted)

        volume_args = {"encrypted": True}
        Volume(self, **volume_args)

class SimpleDictValue(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        volume_args = {"encrypted": False}
        Volume(self, **volume_args) # Noncompliant

class CircularAssignment(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        a = b
        b = a
        volume_args = {"encrypted": b}
        Volume(self, **volume_args)

class DictValueLinkedToVariable(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        a = False
#       ^^^^^^^^^> 1 {{Propagated setting.}}
        volume_args = {"encrypted": a}
        Volume(self, **volume_args) # Noncompliant
#                    ^^^^^^^^^^^^^ 1

class DictValueLinkedToVariableLinkedToVariable(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        b = False
#       ^^^^^^^^^> 1 {{Propagated setting.}}
        a = b
        volume_args = {"encrypted": a}
        Volume(self, **volume_args) # Noncompliant {{Make sure that using unencrypted volumes is safe here.}}
#                    ^^^^^^^^^^^^^ 1

class DictValueLinkedToDictValue(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        a = {"b": False}
        volume_args = {"encrypted": a.b}
        Volume(self, **volume_args)

class DictValueLinkedToUnknownVariable(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        a = b
        volume_args = {"encrypted": a}
        Volume(self, **volume_args)
