from anyio.abc import TaskStatus as DirectStatus
from genericDefaultsExported import TaskStatus as ReexportedStatus, Required

direct: DirectStatus
reexported: ReexportedStatus
nested: list[ReexportedStatus]

def status(arg: ReexportedStatus) -> ReexportedStatus:
    ...

required: Required  # Noncompliant
explicit: Required[int]
