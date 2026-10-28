from _typeshed import Incomplete
from typing import Any
from typing_extensions import Self

from SonarPythonAnalyzerFakeStub import CustomStubBase

class argon2(CustomStubBase):
    @classmethod
    def using(
        cls,
        type: Incomplete | None = ...,
        memory_cost: Incomplete | None = ...,
        salt_len: Incomplete | None = ...,
        time_cost: Incomplete | None = ...,
        digest_size: Incomplete | None = ...,
        checksum_size: Incomplete | None = ...,
        hash_len: Incomplete | None = ...,
        max_threads: Incomplete | None = ...,
        **kwds: Any,
    ) -> type[Self]: ...
