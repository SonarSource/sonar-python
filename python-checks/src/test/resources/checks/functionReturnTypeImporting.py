from typing import NamedTuple
from functionReturnTypeImported import ImportedFieldMembersOnlyNamedTuple, ImportedMethodMembersOnlyNamedTuple

def get_imported_field_members_only_named_tuple() -> ImportedFieldMembersOnlyNamedTuple:
  return None # FN SONARPY-2316

def get_imported_method_members_only_named_tuple() -> ImportedMethodMembersOnlyNamedTuple:
  return None # Noncompliant

class LocallyDefinedFieldMembersOnlyNamedTuple(NamedTuple):
  id: str

def get_locally_defined_field_members_only_named_tuple() -> LocallyDefinedFieldMembersOnlyNamedTuple:
  return None # Noncompliant
