from nonCallableCalledImported import MyNonCallableClass, MyCallableClass
import nonCallableCalledImported

imported_non_callable = MyNonCallableClass()
imported_non_callable_alias = nonCallableCalledImported.MyNonCallableClassAlias()
imported_callable = MyCallableClass()

imported_non_callable()  # Noncompliant
imported_non_callable_alias()  # Noncompliant
imported_callable()


class LocallyDefinedNonCallable:
    ...

local_non_callable = LocallyDefinedNonCallable()
local_non_callable()  # Noncompliant
