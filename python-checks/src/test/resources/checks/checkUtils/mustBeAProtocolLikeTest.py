from typing import Protocol
import typing
from zope.interface import Interface
import zope
from some_module import Unknown

class ProtocolLike01(Protocol):
    ...

class ProtocolLike02(typing.Protocol):
    ...

class ProtocolLike03(Interface):
    ...

class ProtocolLike04(zope.interface.Interface):
    ...

class ProtocolLike05(Protocol, Interface):
    ...

class ProtocolLike06(Protocol, Interface):
    ...

class NonProtocolLike01:
    ...

class NonProtocolLike02(Unknown):
    ...

class NonProtocolLike03(Protocol):
    ...

NonProtocolLike03 = 42
