from changeMethodContractParent import ParentClass
from typing import Any

class IntermediateClass(ParentClass):
  ...

class IntermediateWithAnyClass(ParentClass[Any]):
  ...
