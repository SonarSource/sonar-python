
def class_with_generic_in_declaration():
    from typing import Generic, TypeVar
    _T_co = TypeVar("_T_co", covariant=True, bound=str)
    class ClassA(Generic[_T_co]):  # Noncompliant {{Use the "type" parameter syntax to declare this generic class.}}
        # ^^^^^^ ^^^^^^^^^^^^^^<                  {{"Generic" parent.}}
        ...

def class_with_typing_generic_in_declaration():
    import typing
    _T_co = typing.TypeVar("_T_co", covariant=True, bound=str)
    class ClassA(typing.Generic[_T_co]):  # Noncompliant {{Use the "type" parameter syntax to declare this generic class.}}
        # ^^^^^^ ^^^^^^^^^^^^^^^^^^^^^<                  {{"Generic" parent.}}
        ...


def class_with_generic_from_variable():
    from typing import Generic, TypeVar
    _T_co = TypeVar("_T_co", covariant=True, bound=str)
    generic = Generic[_T_co]
    #         ^^^^^^^^^^^^^^>              {{"Generic" is assigned here.}}
    class ClassA(generic):  # Noncompliant {{Use the "type" parameter syntax to declare this generic class.}}
        # ^^^^^^ ^^^^^^^<                  {{"Generic" parent.}}
        ...

def class_with_generic_without_subscription():
    from typing import Generic
    class ClassA(Generic):  # Noncompliant {{Use the "type" parameter syntax to declare this generic class.}}
        # ^^^^^^ ^^^^^^^<                  {{"Generic" parent.}}
        ...

def class_with_generic_multiple_parents(xx):
    from typing import Generic, TypeVar
    _T_co = TypeVar("_T_co", covariant=True, bound=str)
    class ClassA(xx, Generic[_T_co]):  # Noncompliant {{Use the "type" parameter syntax to declare this generic class.}}
        # ^^^^^^     ^^^^^^^^^^^^^^<                  {{"Generic" parent.}}
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

    class Generic:
        ...

    class ClassWithDifferentGenericParent(Generic):
        ...

