import aws_cdk.aws_kinesis as kinesis


class CfnStreamStack(Stack):
    def __init__(self, app: App, id: str) -> None:

        # Noncompliant@+1 {{Make sure that disabling stream encryption is safe here.}}
        kinesis.CfnStream(stream_encryption=None)
        # Noncompliant@+1 {{Omitting `stream_encryption` causes stream encryption to be disabled. Make sure it is safe here.}}
        kinesis.CfnStream()

        kinesis.CfnStream(
            stream_encryption=kinesis.CfnStream.StreamEncryptionProperty(
                encryption_type="KMS",
                key_id="alias/aws/kinesis"
            )
        )

        kinesis.CfnStream(
            stream_encryption={
                "encryptionType": "KMS",
                "keyId": "alias/aws/kinesis"
            }
        )


class StreamStack(Stack):
    def __init__(self, app: App, id: str) -> None:

        # Noncompliant@+1 {{Make sure that disabling stream encryption is safe here.}}
        kinesis.Stream(encryption=kinesis.StreamEncryption.UNENCRYPTED)

        kinesis.Stream()
        kinesis.Stream(encryption=kinesis.StreamEncryption.MANAGED)
