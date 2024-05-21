from nonCallableCalledImported import MyNonCallableClass, MyCallableClass


imported_non_callable = MyNonCallableClass()
imported_callable = MyCallableClass()

imported_non_callable()  # Noncompliant
imported_callable()


class LocallyDefinedNonCallable:
    ...

local_non_callable = LocallyDefinedNonCallable()
local_non_callable()  # Noncompliant
