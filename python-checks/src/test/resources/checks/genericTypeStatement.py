from typing import TypeVar


def non_compliant_1():
    _T = TypeVar("_T")
    #    ^^^^^^^^^^^^^>       {{"TypeVar" is assigned here.}}
    type MyAlias = set[_T]  # Noncompliant {{Use a generic type parameter instead of a "TypeVar" in this type statement.}}
    #    ^^^^^^^       ^^<    {{Use of "TypeVar" here.}}

    _P = TypeVar("_P")
    #    ^^^^^^^^^^^^^>            {{"TypeVar" is assigned here.}}
    type MyAlias = dict[_P, _S]  # Noncompliant
    #    ^^^^^^^        ^^<        {{Use of "TypeVar" here.}}
    #                       ^^@-1< {{Use of "TypeVar" here.}}
    _S = TypeVar("_S")
    #    ^^^^^^^^^^^^^<            {{"TypeVar" is assigned here.}}

    _R = TypeVar("_R")
    #    ^^^^^^^^^^^^^>              {{"TypeVar" is assigned here.}}
    type MyAlias[T] = dict[T, _R]  # Noncompliant
    #    ^^^^^^^              ^^<    {{Use of "TypeVar" here.}}

    M = TypeVar("M")
    #   ^^^^^^^^^^^^>           {{"TypeVar" is assigned here.}}
    type MyAlias[M] = set[M]  # Noncompliant
    #    ^^^^^^^          ^<    {{Use of "TypeVar" here.}}


def compliant(AType):
    type MyAlias[T] = dict[T, str]
    type MyAlias = set[str]
    type MyAlias = set[AType]
