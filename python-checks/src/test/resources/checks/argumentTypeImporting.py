from argumentTypeImported import imported_function

def local_function(params: list[str]): ...

def imported_fn():
  imported_function(("Name", "Tags")) # Noncompliant
  local_function(("Name", "Tags")) # Noncompliant
