from submodule import SubmoduleException
from subpackage.parent import ParentException
from subpackage import SubpackageException
from relative_imports.module import RelativeImportException
from relative_imports.sub.inner import RelativeImportSubpackageException

try:
    raise SubmoduleException()
except (SubmoduleException, NotImplementedError): # Noncompliant
    print("Foo")

try:
    raise SubmoduleException()
except (SubmoduleException, ValueError): # OK
    print("Foo")

try:
    raise ParentException()
except (ParentException, NotImplementedError): # Noncompliant
    print("Foo")

try:
    raise ParentException()
except (ParentException, ValueError): # OK
    print("Foo")

try:
    raise SubpackageException()
except (SubpackageException, NotImplementedError): # Noncompliant
    print("Foo")

try:
    raise SubpackageException()
except (SubpackageException, ValueError): # OK
    print("Foo")

try:
    raise RelativeImportException()
except (RelativeImportException, NotImplementedError): # Noncompliant
    print("Foo")

try:
    raise RelativeImportException()
except (RelativeImportException, ValueError): # OK
    print("Foo")

try:
    raise RelativeImportSubpackageException()
except (RelativeImportSubpackageException, NotImplementedError): # Noncompliant
    print("Foo")

try:
    raise RelativeImportSubpackageException()
except (RelativeImportSubpackageException, ValueError): # OK
    print("Foo")
