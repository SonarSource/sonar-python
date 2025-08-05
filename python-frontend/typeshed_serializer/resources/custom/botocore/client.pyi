from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import Any, Optional

class BaseClient(CustomStubBase):
    def invoke(self, FunctionName: str, InvocationType: str, Payload: str) -> Any: ...

    def list_objects_v2(self, Bucket: str,
                        Delimiter: Optional[str] = ...,
                        EncodingType: Optional[Any] = ...,
                        MaxKeys: Optional[int] = ...,
                        Prefix: Optional[str] = ...,
                        ContinuationToken: Optional[str] = ...,
                        FetchOwner: Optional[bool] = ...,
                        StartAfter: Optional[str] = ...,
                        RequestPayer: Optional[Any] = ...,
                        ExpectedBucketOwner: Optional[str] = ...,
                        OptionalObjectAttributes: Optional[list[Any]] = ...) -> Any: ...
    
    def scan(self, TableName: str,
             IndexName: str = ...,
             AttributesToGet: list[str] = ...,
             Limit: int = ...,
             Select: str = ...,
             ScanFilter: Any = ...,
             ConditionalOperator: str = ...,
             ExclusiveStartKey: Any = ...,
             ReturnConsumedCapacity: str = ...,
             TotalSegments: int = ...,
             Segment: int = ...,
             ProjectionExpression: str = ...,
             FilterExpression: str = ...,
             ExpressionAttributeNames: Any = ...,
             ExpressionAttributeValues: Any = ...,
             ConsistentRead: bool = ...) -> Any: ...
