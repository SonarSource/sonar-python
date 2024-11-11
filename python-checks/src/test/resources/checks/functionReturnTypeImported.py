from typing import NamedTuple

class ImportedFieldMembersOnlyNamedTuple(NamedTuple):
  id: str

class ImportedMethodMembersOnlyNamedTuple(NamedTuple):
  def id(self): ...
