
def class_with_generic_in_declaration():
    from typing import Generic, TypeVar
    _T_co = TypeVar("_T_co", covariant=True, bound=str)
    class ClassA(Generic[_T_co]):  # Noncompliant {{Use the type parameter syntax to declare this generic class.}}
        # ^^^^^^ ^^^^^^^^^^^^^^<
        ...

def class_with_typing_generic_in_declaration():
    import typing
    _T_co = typing.TypeVar("_T_co", covariant=True, bound=str)
    class ClassA(typing.Generic[_T_co]):  # Noncompliant {{Use the type parameter syntax to declare this generic class.}}
        # ^^^^^^ ^^^^^^^^^^^^^^^^^^^^^<
        ...


def class_with_generic_from_variable():
    from typing import Generic, TypeVar
    _T_co = TypeVar("_T_co", covariant=True, bound=str)
    generic = Generic[_T_co]
    #         ^^^^^^^^^^^^^^>
    class ClassA(generic):  # Noncompliant {{Use the type parameter syntax to declare this generic class.}}
        # ^^^^^^ ^^^^^^^<
        ...

def class_with_generic_without_subscription():
    from typing import Generic
    class ClassA(Generic):  # Noncompliant {{Use the type parameter syntax to declare this generic class.}}
        # ^^^^^^ ^^^^^^^<
        ...

def class_with_generic_multiple_parents(xx):
    from typing import Generic, TypeVar
    _T_co = TypeVar("_T_co", covariant=True, bound=str)
    class ClassA(xx, Generic[_T_co]):  # Noncompliant {{Use the type parameter syntax to declare this generic class.}}
        # ^^^^^^     ^^^^^^^^^^^^^^<
        ...

def compliants(xx):
    class ClassWithNewTypeParameterSyntax[T: str]:
        ...

    class ClassWithArbitraryParent(xx):
        ...

    args = xx
    class ClassWithUnpackingArgument(*args):
        ...

    def foo():
        ...
    class ClassWithFunctionCallParent(foo()):
        ...

    generic = 42
    class ClassWithUnrelatedAssignedValueParent(generic):
        ...


def compliant_3(xx):
    # Test case where we create a class called Generic, which coincidentally is a subscription expression,
    # but is not the Generic class that we are interested in.
    pass


def compliant_4(xx):
    # Test case where we try to instantiate Generic without the subscription check.
    pass


# Define a class called Generic from a separate module.
